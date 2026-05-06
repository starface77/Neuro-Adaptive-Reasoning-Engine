"""Docker-based sandbox for secure code execution.

Supports multiple languages: Python, Node.js, Rust, Go, Java, C++
Provides isolation, resource limits, and timeout enforcement.
"""

import docker
import tempfile
import os
import time
from nare.utils.logger import get_logger
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

log = get_logger("nare.execution.sandboxes.base.base.docker")

class DockerSandbox:
    """Secure Docker-based code execution sandbox."""

    LANGUAGE_CONFIGS = {
        'python': {
            'image': 'python:3.11-slim',
            'file_ext': '.py',
            'run_cmd': 'python {file}',
            'install_cmd': 'pip install {packages}',
        },
        'node': {
            'image': 'node:20-slim',
            'file_ext': '.js',
            'run_cmd': 'node {file}',
            'install_cmd': 'npm install -g {packages}',
        },
        'rust': {
            'image': 'rust:1.75-slim',
            'file_ext': '.rs',
            'run_cmd': 'rustc {file} -o /tmp/prog && /tmp/prog',
            'install_cmd': 'cargo install {packages}',
        },
        'go': {
            'image': 'golang:1.21-alpine',
            'file_ext': '.go',
            'run_cmd': 'go run {file}',
            'install_cmd': 'go install {packages}',
        },
        'java': {
            'image': 'openjdk:17-slim',
            'file_ext': '.java',
            'run_cmd': 'javac {file} && java Main',
            'install_cmd': None,
        },
        'cpp': {
            'image': 'gcc:13-slim',
            'file_ext': '.cpp',
            'run_cmd': 'g++ {file} -o /tmp/prog && /tmp/prog',
            'install_cmd': 'apt-get update && apt-get install -y {packages}',
        },
    }

    def __init__(
        self,
        language: str = 'python',
        memory_limit: str = '512m',
        cpu_limit: float = 1.0,
        timeout: int = 30,
        network_disabled: bool = True,
    ):
        """Initialize Docker sandbox.

        Args:
            language: Programming language (python, node, rust, go, java, cpp)
            memory_limit: Memory limit (e.g., '512m', '1g')
            cpu_limit: CPU limit (fraction of CPU, e.g., 1.0 = 1 core)
            timeout: Execution timeout in seconds
            network_disabled: Disable network access for security
        """
        self.language = language.lower()
        if self.language not in self.LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")

        self.config = self.LANGUAGE_CONFIGS[self.language]
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.network_disabled = network_disabled

        try:
            self.client = docker.from_env()

            self._ensure_image()
        except Exception as e:
            log.error(f"Failed to initialize Docker client: {e}")
            raise

    def _ensure_image(self) -> None:
        """Ensure Docker image is available."""
        image_name = self.config['image']
        try:
            self.client.images.get(image_name)
            log.info(f"Docker image {image_name} already available")
        except docker.errors.ImageNotFound:
            log.info(f"Pulling Docker image {image_name}...")
            self.client.images.pull(image_name)
            log.info(f"Successfully pulled {image_name}")

    def execute(
        self,
        code: str,
        stdin: Optional[str] = None,
        packages: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Execute code in Docker sandbox.

        Args:
            code: Source code to execute
            stdin: Optional stdin input
            packages: Optional list of packages to install

        Returns:
            Dict with keys: stdout, stderr, exit_code, elapsed, error
        """
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            code_file = tmpdir_path / f"main{self.config['file_ext']}"
            code_file.write_text(code, encoding='utf-8')

            stdin_file = None
            if stdin:
                stdin_file = tmpdir_path / "stdin.txt"
                stdin_file.write_text(stdin, encoding='utf-8')

            try:

                if packages and self.config['install_cmd']:
                    install_result = self._install_packages(tmpdir, packages)
                    if install_result['exit_code'] != 0:
                        return {
                            'stdout': install_result['stdout'],
                            'stderr': f"Package installation failed:\n{install_result['stderr']}",
                            'exit_code': install_result['exit_code'],
                            'elapsed': time.time() - start_time,
                            'error': 'package_install_failed',
                        }

                result = self._run_container(tmpdir, code_file.name, stdin_file)
                result['elapsed'] = time.time() - start_time
                return result

            except docker.errors.ContainerError as e:
                return {
                    'stdout': e.stdout.decode('utf-8') if e.stdout else '',
                    'stderr': e.stderr.decode('utf-8') if e.stderr else str(e),
                    'exit_code': e.exit_status,
                    'elapsed': time.time() - start_time,
                    'error': 'container_error',
                }
            except docker.errors.APIError as e:
                return {
                    'stdout': '',
                    'stderr': f"Docker API error: {e}",
                    'exit_code': -1,
                    'elapsed': time.time() - start_time,
                    'error': 'docker_api_error',
                }
            except Exception as e:
                return {
                    'stdout': '',
                    'stderr': f"Unexpected error: {e}",
                    'exit_code': -1,
                    'elapsed': time.time() - start_time,
                    'error': 'unexpected_error',
                }

    def _install_packages(self, tmpdir: str, packages: list) -> Dict[str, Any]:
        """Install packages in container.

        Args:
            tmpdir: Temporary directory path
            packages: List of package names

        Returns:
            Dict with stdout, stderr, exit_code
        """
        install_cmd = self.config['install_cmd'].format(packages=' '.join(packages))

        container = self.client.containers.run(
            self.config['image'],
            command=f"sh -c '{install_cmd}'",
            volumes={tmpdir: {'bind': '/workspace', 'mode': 'rw'}},
            working_dir='/workspace',
            mem_limit=self.memory_limit,
            nano_cpus=int(self.cpu_limit * 1e9),
            network_disabled=self.network_disabled,
            detach=True,
            remove=False,
        )

        try:
            result = container.wait(timeout=self.timeout)
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')

            return {
                'stdout': logs,
                'stderr': '',
                'exit_code': result['StatusCode'],
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
            }
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def _run_container(
        self,
        tmpdir: str,
        code_filename: str,
        stdin_file: Optional[Path],
    ) -> Dict[str, Any]:
        """Run code in Docker container.

        Args:
            tmpdir: Temporary directory path
            code_filename: Name of code file
            stdin_file: Optional stdin file path

        Returns:
            Dict with stdout, stderr, exit_code
        """
        # Sanitize filename to prevent command injection
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '', code_filename)
        if safe_filename != code_filename:
            log.warning(f"Sanitized filename: {code_filename} -> {safe_filename}")
            code_filename = safe_filename

        run_cmd = self.config['run_cmd'].format(file=code_filename)

        if stdin_file:
            safe_stdin = re.sub(r'[^a-zA-Z0-9._-]', '', stdin_file.name)
            run_cmd = f"{run_cmd} < {safe_stdin}"

        container = self.client.containers.run(
            self.config['image'],
            command=f"sh -c '{run_cmd}'",
            volumes={tmpdir: {'bind': '/workspace', 'mode': 'rw'}},
            working_dir='/workspace',
            mem_limit=self.memory_limit,
            nano_cpus=int(self.cpu_limit * 1e9),
            network_disabled=self.network_disabled,
            detach=True,
            remove=False,
            # Security hardening
            security_opt=['no-new-privileges'],
            cap_drop=['ALL'],
            read_only=True,  # Make root filesystem read-only
            tmpfs={'/tmp': 'size=100m,mode=1777'},  # Allow /tmp writes
            user='nobody',  # Run as non-root user
        )

        try:

            result = container.wait(timeout=self.timeout)
            exit_code = result['StatusCode']

            logs = container.logs(stdout=True, stderr=True).decode('utf-8')

            stderr_keywords = ['error', 'traceback', 'exception', 'warning', 'failed']
            lines = logs.split('\n')
            stderr_lines = [l for l in lines if any(kw in l.lower() for kw in stderr_keywords)]
            stdout_lines = [l for l in lines if l not in stderr_lines]

            return {
                'stdout': '\n'.join(stdout_lines),
                'stderr': '\n'.join(stderr_lines),
                'exit_code': exit_code,
                'error': None if exit_code == 0 else 'execution_failed',
            }

        except docker.errors.ContainerError as e:
            return {
                'stdout': e.stdout.decode('utf-8') if e.stdout else '',
                'stderr': e.stderr.decode('utf-8') if e.stderr else str(e),
                'exit_code': e.exit_status,
                'error': 'container_error',
            }
        except Exception as e:

            try:
                container.kill()
            except Exception:
                pass

            return {
                'stdout': '',
                'stderr': f"Execution timeout or error: {e}",
                'exit_code': -1,
                'error': 'timeout' if 'timeout' in str(e).lower() else 'execution_error',
            }
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def cleanup(self) -> None:
        """Cleanup Docker resources."""
        try:

            containers = self.client.containers.list(
                all=True,
                filters={'status': 'exited'}
            )
            for container in containers:
                if container.image.tags and self.config['image'] in container.image.tags:
                    container.remove()
        except Exception as e:
            log.warning(f"Cleanup failed: {e}")

def execute_code(
    code: str,
    language: str = 'python',
    stdin: Optional[str] = None,
    packages: Optional[list] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Execute code in Docker sandbox (convenience function).

    Args:
        code: Source code
        language: Programming language
        stdin: Optional stdin
        packages: Optional packages to install
        timeout: Timeout in seconds

    Returns:
        Execution result dict
    """
    sandbox = DockerSandbox(language=language, timeout=timeout)
    try:
        return sandbox.execute(code, stdin=stdin, packages=packages)
    finally:
        sandbox.cleanup()
