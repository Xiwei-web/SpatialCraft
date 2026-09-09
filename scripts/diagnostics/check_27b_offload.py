"""Reproduce direct-weight CPU offload failure, then check the repaired provider."""
import argparse
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from spatialcraft.experiments.runtime import ExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.models import ContentPart, RequestBuilder

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
torch.set_num_threads(1)
settings = ExperimentSettings.load(args.project/'configs/experiments/qwen36_27b_omni3d.yaml')
with tempfile.TemporaryDirectory() as tmp:
    runtime = ExperimentRuntime(args.project, Path(tmp), settings, {})
    provider = runtime.local
    # Load the original behavior for the structural reproduction.
    from spatialcraft.models.providers import transformers_local
    original_repair = transformers_local.repair_qwen_cpu_offload
    transformers_local.repair_qwen_cpu_offload = lambda model: []
    model, _ = provider._load()
    transformers_local.repair_qwen_cpu_offload = original_repair
    picture = Path(tmp)/'red.png'
    Image.new('RGB', (224,224), 'red').save(picture)
    request = (RequestBuilder(runtime.model).system('Answer directly without a thinking block.')
               .user('What is the dominant color? Reply with one word.', media=(ContentPart.image_uri(str(picture)),))
               .metadata(chat_template_kwargs={'enable_thinking': False})
               .settings(max_output_tokens=32, temperature=.7, top_p=.9, seed=42).build())
    evidence = {'job_id': os.environ['SLURM_JOB_ID'], 'hostname': os.uname().nodename}
    # Force a second decoding step and intercept missing weights before CUDA is poisoned.
    originals = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Qwen3_5GatedDeltaNet':
            previous = module.causal_conv1d_update
            def checked(*values, _previous=previous, _name=name, **kwargs):
                weight = values[2] if len(values) > 2 else kwargs['weight']
                if weight.device.type == 'meta':
                    raise RuntimeError(f'OFFLOAD_REPRODUCED: {_name}.conv1d.weight is meta during cached decode')
                return _previous(*values, **kwargs)
            module.causal_conv1d_update = checked
            originals.append((module, previous))
    from dataclasses import replace
    forced = replace(request, settings=replace(request.settings, extra={'min_new_tokens': 3}))
    try:
        provider.generate(forced)
    except Exception as exc:
        evidence['original_error'] = str(exc)
        assert 'OFFLOAD_REPRODUCED:' in str(exc), str(exc)
        print(str(exc), flush=True)
    else:
        raise AssertionError('Expected missing convolution weights in original offload code')
    for module, previous in originals:
        module.causal_conv1d_update = previous
    evidence['repaired_modules'] = original_repair(model)
    print(json.dumps(evidence, indent=2), flush=True)
    (args.output/'reproduction.json').write_text(json.dumps(evidence, indent=2))
    # Reuse the loaded model for the complete formal provider acceptance.
    from spatialcraft.experiments.runtime import ConfiguredLocalProvider
    ConfiguredLocalProvider._load = lambda self: (model, provider._processor)
    sys.argv = ['check_experiment_provider.py', '--project', str(args.project), '--config', str(args.project/'configs/experiments/qwen36_27b_omni3d.yaml'), '--report', str(args.output/'provider.json')]
    runpy.run_path(str(args.project/'scripts/inference/check_experiment_provider.py'), run_name='__main__')
    # Multiple-token cached decode must also work after scoring and tool generation.
    replay = [provider.generate(forced) for _ in range(2)]
    assert replay[0].raw['generated_token_ids'] == replay[1].raw['generated_token_ids']
    assert replay[0].usage.output_tokens >= 3
    (args.output/'extended_decode.json').write_text(json.dumps({'status': 'passed', 'response': replay[0].text, 'token_ids': replay[0].raw['generated_token_ids']}, indent=2))
    print('FULL_ACCEPTANCE_PASSED', flush=True)
