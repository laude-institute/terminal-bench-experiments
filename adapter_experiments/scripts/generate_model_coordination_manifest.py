from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml
from dotenv import dotenv_values

from coordination import CoordinationManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / '.env'
_TEMPLATE_PATTERN = re.compile(r'^\$\{([^}:]+)(?::-([^}]*))?\}$')


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    adapter_id: str
    family: str
    agent_name: str
    model_name: str
    display_name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    env_templates: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _job_template(n_attempts: int, n_concurrent_trials: int) -> dict[str, Any]:
    return {
        'jobs_dir': 'jobs',
        'n_attempts': n_attempts,
        'timeout_multiplier': 1.0,
        'orchestrator': {
            'type': 'local',
            'n_concurrent_trials': n_concurrent_trials,
            'quiet': False,
            'retry': {
                'max_retries': 3,
                'exclude_exceptions': [
                    'BadRequestError',
                    'RateLimitError',
                    'AgentTimeoutError',
                    'VerifierTimeoutError',
                    'RewardFileNotFoundError',
                ],
                'wait_multiplier': 1.0,
                'min_wait_sec': 1.0,
                'max_wait_sec': 60.0,
            },
        },
        'environment': {
            'type': 'daytona',
            'force_build': False,
            'delete': True,
        },
    }


def _phase_map(dataset_prefix: str, dataset_version: str) -> dict[str, Any]:
    return {
        'phase2': {
            'percent': 1,
            'dataset_name': f'{dataset_prefix}-phase2',
            'dataset_version': dataset_version,
            'description': f'{dataset_prefix} phase2 (1%).',
        },
        'phase3': {
            'percent': 10,
            'dataset_name': f'{dataset_prefix}-phase3',
            'dataset_version': dataset_version,
            'description': f'{dataset_prefix} phase3 (10% cumulative minus earlier phases).',
        },
        'phase4': {
            'percent': 100,
            'dataset_name': f'{dataset_prefix}-phase4',
            'dataset_version': dataset_version,
            'description': f'{dataset_prefix} phase4 (remaining full set after earlier phases).',
        },
    }


def _catalog(
    codex_version: str,
    gemini_cli_version: str,
    claude_code_version: str,
) -> list[AdapterSpec]:
    return [
        AdapterSpec(
            adapter_id='terminus-2__glm-5',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='zai/glm-5',
            display_name='Terminus 2 / GLM-5',
            metadata={'provider': 'zai'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__deepseek-reasoner',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='deepseek/deepseek-reasoner',
            display_name='Terminus 2 / DeepSeek Reasoner',
            metadata={'provider': 'deepseek'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__minimax-m2.5',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='minimax/MiniMax-M2.5',
            display_name='Terminus 2 / MiniMax M2.5',
            kwargs={'api_base': 'https://api.minimax.chat/v1'},
            metadata={'provider': 'minimax'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__kimi-k2.5',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='moonshot/kimi-k2.5',
            display_name='Terminus 2 / Kimi K2.5',
            kwargs={'temperature': 1},
            metadata={'provider': 'moonshot'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__mimo-v2-pro',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='xiaomi_mimo/mimo-v2-pro',
            display_name='Terminus 2 / Xiaomi Mimo v2 Pro',
            metadata={'provider': 'xiaomi_mimo'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__gpt-5.4',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='openai/gpt-5.4',
            display_name='Terminus 2 / GPT-5.4',
            metadata={'provider': 'openai'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__gpt-5-mini',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='openai/gpt-5-mini',
            display_name='Terminus 2 / GPT-5 Mini',
            metadata={'provider': 'openai'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__gpt-5-nano',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='openai/gpt-5-nano',
            display_name='Terminus 2 / GPT-5 Nano',
            metadata={'provider': 'openai'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__gemini-3.1-pro-preview',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='gemini/gemini-3.1-pro-preview',
            display_name='Terminus 2 / Gemini 3.1 Pro Preview',
            metadata={'provider': 'gemini'},
        ),
        AdapterSpec(
            adapter_id='terminus-2__gemini-3-flash-preview',
            family='terminus-2',
            agent_name='terminus-2',
            model_name='gemini/gemini-3-flash-preview',
            display_name='Terminus 2 / Gemini 3 Flash Preview',
            metadata={'provider': 'gemini'},
        ),
        AdapterSpec(
            adapter_id='codex__gpt-5.4',
            family='codex',
            agent_name='codex',
            model_name='openai/gpt-5.4',
            display_name='Codex / GPT-5.4',
            kwargs={'version': codex_version},
            metadata={'provider': 'openai'},
        ),
        AdapterSpec(
            adapter_id='codex__gpt-5-mini',
            family='codex',
            agent_name='codex',
            model_name='openai/gpt-5-mini',
            display_name='Codex / GPT-5 Mini',
            kwargs={'version': codex_version},
            metadata={'provider': 'openai'},
        ),
        AdapterSpec(
            adapter_id='codex__gpt-5-nano',
            family='codex',
            agent_name='codex',
            model_name='openai/gpt-5-nano',
            display_name='Codex / GPT-5 Nano',
            kwargs={'version': codex_version},
            metadata={'provider': 'openai'},
        ),
        AdapterSpec(
            adapter_id='gemini-cli__gemini-3.1-pro-preview',
            family='gemini-cli',
            agent_name='gemini-cli',
            model_name='gemini/gemini-3.1-pro-preview',
            display_name='Gemini CLI / Gemini 3.1 Pro Preview',
            kwargs={'version': gemini_cli_version},
            metadata={'provider': 'gemini'},
        ),
        AdapterSpec(
            adapter_id='gemini-cli__gemini-3-flash-preview',
            family='gemini-cli',
            agent_name='gemini-cli',
            model_name='gemini/gemini-3-flash-preview',
            display_name='Gemini CLI / Gemini 3 Flash Preview',
            kwargs={'version': gemini_cli_version},
            metadata={'provider': 'gemini'},
        ),
        AdapterSpec(
            adapter_id='claude-code__glm-5',
            family='claude-code',
            agent_name='claude-code',
            model_name='anthropic/glm-5',
            display_name='Claude Code / GLM-5',
            kwargs={'version': claude_code_version},
            env_templates={
                'ANTHROPIC_AUTH_TOKEN': '${ZAI_API_KEY}',
                'ANTHROPIC_BASE_URL': 'https://api.z.ai/api/anthropic',
                'ANTHROPIC_MODEL': 'glm-5',
                'ANTHROPIC_DEFAULT_SONNET_MODEL': 'glm-5',
                'ANTHROPIC_DEFAULT_OPUS_MODEL': 'glm-5',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'glm-5',
                'CLAUDE_CODE_SUBAGENT_MODEL': 'glm-5',
                'API_TIMEOUT_MS': '3000000',
            },
            metadata={'provider': 'zai'},
        ),
        AdapterSpec(
            adapter_id='claude-code__deepseek-chat',
            family='claude-code',
            agent_name='claude-code',
            model_name='anthropic/deepseek-chat',
            display_name='Claude Code / DeepSeek Chat',
            kwargs={'version': claude_code_version},
            env_templates={
                'ANTHROPIC_AUTH_TOKEN': '${DEEPSEEK_API_KEY}',
                'ANTHROPIC_BASE_URL': 'https://api.deepseek.com/anthropic',
                'ANTHROPIC_MODEL': 'deepseek-chat',
                'ANTHROPIC_DEFAULT_SONNET_MODEL': 'deepseek-chat',
                'ANTHROPIC_DEFAULT_OPUS_MODEL': 'deepseek-chat',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'deepseek-chat',
                'CLAUDE_CODE_SUBAGENT_MODEL': 'deepseek-chat',
                'API_TIMEOUT_MS': '600000',
            },
            metadata={'provider': 'deepseek'},
        ),
        AdapterSpec(
            adapter_id='claude-code__minimax-m2.5',
            family='claude-code',
            agent_name='claude-code',
            model_name='anthropic/MiniMax-M2.5',
            display_name='Claude Code / MiniMax M2.5',
            kwargs={'version': claude_code_version},
            env_templates={
                'ANTHROPIC_AUTH_TOKEN': '${MINIMAX_API_KEY}',
                'ANTHROPIC_BASE_URL': 'https://api.minimaxi.com/anthropic',
                'ANTHROPIC_MODEL': 'MiniMax-M2.5',
                'ANTHROPIC_DEFAULT_SONNET_MODEL': 'MiniMax-M2.5',
                'ANTHROPIC_DEFAULT_OPUS_MODEL': 'MiniMax-M2.5',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'MiniMax-M2.5',
                'CLAUDE_CODE_SUBAGENT_MODEL': 'MiniMax-M2.5',
                'API_TIMEOUT_MS': '3000000',
            },
            metadata={'provider': 'minimax'},
        ),
        AdapterSpec(
            adapter_id='claude-code__kimi-k2.5',
            family='claude-code',
            agent_name='claude-code',
            model_name='anthropic/kimi-k2.5',
            display_name='Claude Code / Kimi K2.5',
            kwargs={'version': claude_code_version},
            env_templates={
                'ANTHROPIC_AUTH_TOKEN': '${MOONSHOT_API_KEY}',
                'ANTHROPIC_BASE_URL': 'https://api.moonshot.ai/anthropic',
                'ANTHROPIC_MODEL': 'kimi-k2.5',
                'ANTHROPIC_DEFAULT_SONNET_MODEL': 'kimi-k2.5',
                'ANTHROPIC_DEFAULT_OPUS_MODEL': 'kimi-k2.5',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'kimi-k2.5',
                'CLAUDE_CODE_SUBAGENT_MODEL': 'kimi-k2.5',
                'ENABLE_TOOL_SEARCH': 'false',
                'API_TIMEOUT_MS': '3000000',
            },
            metadata={'provider': 'moonshot'},
        ),
        AdapterSpec(
            adapter_id='claude-code__mimo-v2-pro',
            family='claude-code',
            agent_name='claude-code',
            model_name='anthropic/mimo-v2-pro',
            display_name='Claude Code / Xiaomi Mimo v2 Pro',
            kwargs={'version': claude_code_version},
            env_templates={
                'ANTHROPIC_AUTH_TOKEN': '${XIAOMI_MIMO_API_KEY}',
                'ANTHROPIC_BASE_URL': 'https://api.xiaomimimo.com/anthropic',
                'ANTHROPIC_MODEL': 'mimo-v2-pro',
                'ANTHROPIC_DEFAULT_SONNET_MODEL': 'mimo-v2-pro',
                'ANTHROPIC_DEFAULT_OPUS_MODEL': 'mimo-v2-pro',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'mimo-v2-pro',
                'CLAUDE_CODE_SUBAGENT_MODEL': 'mimo-v2-pro',
                'API_TIMEOUT_MS': '3000000',
            },
            metadata={'provider': 'xiaomi_mimo'},
        ),
    ]


def _resolve_template(value: str, env_values: dict[str, str]) -> str:
    match = _TEMPLATE_PATTERN.fullmatch(value)
    if match is None:
        return value

    var_name = match.group(1)
    default = match.group(2)
    resolved = env_values.get(var_name)
    if resolved not in (None, ''):
        return resolved
    if default is not None:
        return default
    raise KeyError(f'missing required environment variable: {var_name}')



def _resolve_obj(obj: Any, env_values: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {key: _resolve_obj(value, env_values) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_resolve_obj(value, env_values) for value in obj]
    if isinstance(obj, str):
        return _resolve_template(obj, env_values)
    return obj



def _load_env_values(env_file: Path) -> dict[str, str]:
    if not env_file.exists():
        raise FileNotFoundError(f'env file not found: {env_file}')

    merged: dict[str, str] = {}
    for key, value in dotenv_values(env_file).items():
        if key is None or value is None:
            continue
        merged[key] = value
    for key, value in os.environ.items():
        merged[key] = value
    return merged



def _select_specs(
    specs: list[AdapterSpec],
    families: set[str] | None,
    adapter_ids: set[str] | None,
) -> list[AdapterSpec]:
    selected = []
    known_ids = {spec.adapter_id for spec in specs}
    if adapter_ids is not None:
        unknown = sorted(adapter_ids - known_ids)
        if unknown:
            raise ValueError(f'unknown adapter ids: {", ".join(unknown)}')

    for spec in specs:
        if families is not None and spec.family not in families:
            continue
        if adapter_ids is not None and spec.adapter_id not in adapter_ids:
            continue
        selected.append(spec)

    if not selected:
        raise ValueError('no adapters were selected')
    return selected



def _assign_contributors(
    specs: list[AdapterSpec],
    contributors: list[str],
) -> dict[str, list[str]]:
    assignments = {name: [] for name in contributors}
    for index, spec in enumerate(specs):
        contributor = contributors[index % len(contributors)]
        assignments[contributor].append(spec.adapter_id)
    return assignments



def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, '/')



def _build_manifest(
    experiment_name: str,
    manifest_path: Path,
    task_set_registry_path: Path,
    task_set_lock_path: Path,
    dataset_prefix: str,
    dataset_version: str,
    registry_name: str,
    n_attempts: int,
    n_concurrent_trials: int,
    specs: list[AdapterSpec],
    contributors: list[str],
    env_values: dict[str, str],
) -> dict[str, Any]:
    manifest_dir = manifest_path.parent
    adapters: dict[str, Any] = {}
    for spec in specs:
        agent = {
            'name': spec.agent_name,
            'model_name': spec.model_name,
        }
        if spec.kwargs:
            agent['kwargs'] = dict(spec.kwargs)
        if spec.env_templates:
            agent['env'] = _resolve_obj(spec.env_templates, env_values)

        metadata = {
            'display_name': spec.display_name,
            'runner': spec.agent_name,
            'family': spec.family,
            'model_name': spec.model_name,
            'generated_by': 'generate_model_coordination_manifest.py',
        }
        metadata.update(spec.metadata)
        adapters[spec.adapter_id] = {
            'agent': agent,
            'metadata': metadata,
        }

    contributor_assignments = _assign_contributors(specs, contributors)
    return {
        'experiment': {
            'name': experiment_name,
            'output_root': '.',
            'registry_name': registry_name,
        },
        'task_set': {
            'registry_path': _relative_path(task_set_registry_path, manifest_dir),
            'lock_path': _relative_path(task_set_lock_path, manifest_dir),
        },
        'phases': _phase_map(dataset_prefix, dataset_version),
        'job_template': _job_template(
            n_attempts=n_attempts,
            n_concurrent_trials=n_concurrent_trials,
        ),
        'adapters': adapters,
        'contributors': {
            name: {'adapters': adapter_ids}
            for name, adapter_ids in contributor_assignments.items()
        },
    }



def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Generate a coordination manifest for the hello-world model matrix. '
            'The script reads API keys from .env and writes a rendered manifest '
            'that plugs directly into adapter_experiments/scripts/coordination.py.'
        )
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='Experiment name used in the manifest and default output path.',
    )
    parser.add_argument(
        '--task-set-registry-path',
        type=Path,
        required=True,
        help='Unified task-set registry JSON produced by sample_registry.py.',
    )
    parser.add_argument(
        '--task-set-lock-path',
        type=Path,
        required=True,
        help='Unified task-set lock JSON produced by sample_registry.py.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'Path to write the rendered coordination manifest. Defaults to '
            'outputs/adapter_experiments/<experiment-name>/coordination.generated.yaml.'
        ),
    )
    parser.add_argument(
        '--env-file',
        type=Path,
        default=DEFAULT_ENV_FILE,
        help='Path to the .env file to read API keys from.',
    )
    parser.add_argument(
        '--dataset-prefix',
        type=str,
        default=None,
        help='Prefix used for generated phase dataset names. Defaults to adapter-experiments-<experiment-name>.',
    )
    parser.add_argument(
        '--dataset-version',
        type=str,
        default='1.0',
        help='Dataset version used for the generated phase datasets.',
    )
    parser.add_argument(
        '--registry-name',
        type=str,
        default='local',
        help='Registry name written into generated dataset configs.',
    )
    parser.add_argument(
        '--contributor',
        action='append',
        default=None,
        help='Contributor name. Repeat to assign adapters round-robin across multiple contributors. Defaults to local.',
    )
    parser.add_argument(
        '--family',
        action='append',
        choices=['terminus-2', 'codex', 'gemini-cli', 'claude-code'],
        default=None,
        help='Optional family filter. Repeat to include multiple families.',
    )
    parser.add_argument(
        '--adapter-id',
        action='append',
        default=None,
        help='Optional adapter id filter. Repeat to include multiple specific adapters.',
    )
    parser.add_argument(
        '--n-attempts',
        type=int,
        default=5,
        help='n_attempts value to place in job_template.',
    )
    parser.add_argument(
        '--n-concurrent-trials',
        type=int,
        default=32,
        help='orchestrator.n_concurrent_trials value to place in job_template.',
    )
    parser.add_argument(
        '--codex-version',
        type=str,
        default='0.116.0',
        help='Codex CLI version pin for generated codex adapters.',
    )
    parser.add_argument(
        '--gemini-cli-version',
        type=str,
        default='0.11.3',
        help='Gemini CLI version pin for generated gemini-cli adapters.',
    )
    parser.add_argument(
        '--claude-code-version',
        type=str,
        default='2.0.31',
        help='Claude Code version pin for generated claude-code adapters.',
    )
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        task_set_registry_path = args.task_set_registry_path.expanduser().resolve()
        task_set_lock_path = args.task_set_lock_path.expanduser().resolve()
        if not task_set_registry_path.exists():
            raise FileNotFoundError(f'task set registry not found: {task_set_registry_path}')
        if not task_set_lock_path.exists():
            raise FileNotFoundError(f'task set lock not found: {task_set_lock_path}')

        output_path = args.output
        if output_path is None:
            output_path = (
                REPO_ROOT
                / 'outputs'
                / 'adapter_experiments'
                / args.experiment_name
                / 'coordination.generated.yaml'
            )
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        env_values = _load_env_values(args.env_file.expanduser().resolve())
        specs = _catalog(
            codex_version=args.codex_version,
            gemini_cli_version=args.gemini_cli_version,
            claude_code_version=args.claude_code_version,
        )
        selected_specs = _select_specs(
            specs=specs,
            families=set(args.family) if args.family else None,
            adapter_ids=set(args.adapter_id) if args.adapter_id else None,
        )
        contributors = args.contributor or ['local']
        dataset_prefix = args.dataset_prefix or f'adapter-experiments-{args.experiment_name}'

        manifest = _build_manifest(
            experiment_name=args.experiment_name,
            manifest_path=output_path,
            task_set_registry_path=task_set_registry_path,
            task_set_lock_path=task_set_lock_path,
            dataset_prefix=dataset_prefix,
            dataset_version=args.dataset_version,
            registry_name=args.registry_name,
            n_attempts=args.n_attempts,
            n_concurrent_trials=args.n_concurrent_trials,
            specs=selected_specs,
            contributors=contributors,
            env_values=env_values,
        )

        CoordinationManifest.model_validate(manifest)
        output_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False))

        print(f'Wrote manifest: {output_path}')
        print(f'Adapters: {len(selected_specs)}')
        for contributor_name, payload in manifest['contributors'].items():
            print(f'  - {contributor_name}: {len(payload["adapters"])} adapters')
        print('Next steps:')
        print(
            '  '
            'uv run python adapter_experiments/scripts/coordination.py '
            f'show --manifest {shlex.quote(str(output_path))}'
        )
        first_contributor = contributors[0]
        print(
            '  '
            'uv run python adapter_experiments/scripts/coordination.py '
            f'contributor --manifest {shlex.quote(str(output_path))} '
            f'--phase phase2 --contributor {shlex.quote(first_contributor)}'
        )
        return 0
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
