from utils import get_sb_environment
import subprocess

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
     
    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        try:
            output = self.env.execute(command)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ValueError(output)
        except TimeoutError:
            raise ValueError("TimeoutError")
        return output['output']
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            output = self.env.execute("git add -A && git diff --cached")
            patch_output = output['output']
            if patch_output.strip():
                patch_output = patch_output.strip()
                output['output'] = patch_output
                return patch_output
            else:
                return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"{result}\n\nError running git commands: {e}"
    
    # -------------------- TODO(student): add more functions here if you want --------------------
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        Replace the content of the file from the given line to the given line with the given content.
        Line numbers are 1-indexed. The content will REPLACE the specified lines (from_line to to_line inclusive).
        
        CRITICAL: Preserve indentation exactly as it appears in the surrounding code. 
        Read the file first to see the exact indentation, then match it in your content.
        
        IMPORTANT: After calling this function, the file's line numbers change. You MUST re-read the file
        to see the new line numbers before making any further edits.
        
        Args:
            file_path (str): path to the file to edit
            from_line (int): starting line number (1-indexed, inclusive) 
            to_line (int): ending line number (1-indexed, inclusive)
            content (str): new content to replace the lines (must preserve exact indentation)
        
        Returns:
            Success message with new line count, or error description
        """
        try:
            # Read the file
            read_output_d = self.env.execute(f"cat {file_path}")
            read_output = read_output_d['output']
            lines = read_output.split('\n')

            try:
                from_line = int(from_line)
            except ValueError:
                return f"Error: Invalid from_line. from_line={from_line}. It must be a valid integer."
            try:
                to_line = int(to_line)
            except ValueError:
                return f"Error: Invalid to_line. to_line={to_line}. It must be a valid integer."
            
            # Validate line numbers
            if from_line < 1 or to_line < 1 or from_line > to_line:
                return f"Error: Invalid line numbers. from_line={from_line}, to_line={to_line}"
            
            if to_line > len(lines):
                return f"Error: to_line ({to_line}) exceeds file length ({len(lines)})"
            
            # Replace lines (convert to 0-indexed)
            # Split content into lines to maintain structure
            content_lines = content.split('\n')
            new_lines = lines[:from_line-1] + content_lines + lines[to_line:]
            new_content = '\n'.join(new_lines)
            
            # Write to temporary file on HOST to avoid argument list too long errors
            import tempfile
            import os
            
            # Create temp file on host system
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as host_temp:
                host_temp.write(new_content)
                host_temp_path = host_temp.name
            
            container_temp_path = '/tmp/swe_agent_temp_edit.txt'
            
            try:
                # Use stdin piping to transfer file into container (avoids command-line size limits and docker cp permission issues)
                container_id = getattr(self.env, 'container_id', None)
                if not container_id:
                    raise ValueError("Container ID not available - this method requires DockerEnvironment")
                
                docker_executable = getattr(self.env.config, 'executable', 'docker')
                
                # Use docker exec with stdin to pipe content into container
                # This avoids docker cp UID/GID issues and works with any file size
                with open(host_temp_path, 'rb') as f:
                    file_content = f.read()
                
                # Use docker exec with stdin to write the file
                write_cmd = [
                    docker_executable, 'exec', '-i', container_id,
                    'bash', '-c', f'cat > {container_temp_path}'
                ]
                result = subprocess.run(
                    write_cmd,
                    input=file_content,
                    capture_output=True,
                    timeout=300  # 5 minute timeout for large files
                )
                
                if result.returncode != 0:
                    file_size = os.path.getsize(host_temp_path)
                    error_msg = f"Failed to write file to container. File size: {file_size} bytes. "
                    if result.stderr:
                        error_msg += f"Error: {result.stderr.decode('utf-8', errors='replace')}"
                    if result.stdout:
                        error_msg += f" Output: {result.stdout.decode('utf-8', errors='replace')}"
                    raise RuntimeError(error_msg)
                
                # Move temp file to target location in container
                self.env.execute(f"cp {container_temp_path} {file_path}")
                self.env.execute(f"rm -f {container_temp_path}")

                lines_removed = to_line - from_line + 1
                lines_added = len(content_lines)
                net_change = lines_added - lines_removed
                
                msg = (
                    f"✓ Successfully replaced lines {from_line}-{to_line} in {file_path}. "
                    f"Replaced {lines_removed} lines with {lines_added} lines (net change: {net_change:+d} lines).\n"
                    f"⚠️  IMPORTANT: Line numbers have changed! You must re-read the file before the next edit.\n"
                    f"The new file has approximately {len(lines) + net_change} total lines."
                )
                return self._append_syntax_warning_if_needed(file_path, msg)
            
            finally:
                # Always clean up host temp file
                try:
                    os.unlink(host_temp_path)
                except Exception:
                    pass
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in replace_in_file: {str(e)}"
    
    def show_file(self, file_path: str, start_line: "int | None" = None, end_line: "int | None" = None) -> str:
        """
        Show the content of the file with line numbers.
        
        Args:
            file_path (str): path to the file to show
            start_line (int, optional): starting line number (1-indexed)
            end_line (int, optional): ending line number (1-indexed)
        
        Returns:
            The content of the file with line numbers
        """
        try:
            if start_line is not None and end_line is not None:
                # Show specific line range
                cmd = f"sed -n '{start_line},{end_line}p' {file_path} | nl -v {start_line} -w 6 -s '  '"
            elif start_line is not None:
                # Show from start_line to end
                cmd = f"sed -n '{start_line},$p' {file_path} | nl -v {start_line} -w 6 -s '  '"
            else:
                # Show entire file
                cmd = f"cat -n {file_path}"
            
            output = self.env.execute(cmd)
            return output['output']
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error reading file: {str(e)}"
    
    def search_in_file(self, file_path: str, pattern: str, use_regex: bool = True) -> str:
        """
        Search for a pattern in a file and return matching lines with line numbers.
        
        Args:
            file_path (str): path to the file to search
            pattern (str): pattern to search for
            use_regex (bool): if False, treat the pattern as a fixed string
        
        Returns:
            Matching lines with line numbers
        """
        try:
            esc = pattern.replace("'", "'\\''")
            fixed_flag = "-F" if not use_regex else ""
            cmd = f"grep {fixed_flag} -n '{esc}' {file_path}"
            output = self.env.execute(cmd)
            return output['output']
        except Exception as e:
            import traceback
            traceback.print_exc()
            # grep returns non-zero when no matches found
            return f"No matches found or error: {str(e)}"
    
    def list_files(self, directory: str = ".") -> str:
        """
        List files in a directory.
        
        Args:
            directory (str): path to directory (default: current directory)
        
        Returns:
            List of files and directories
        """
        try:
            cmd = f"ls -la {directory}"
            output = self.env.execute(cmd)
            return output['output']
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error listing directory: {str(e)}"
    
    def find_file(self, filename: str, directory: str = ".") -> str:
        """
        Find files matching a pattern in a directory tree.
        
        Args:
            filename (str): filename pattern to search for
            directory (str): starting directory (default: current directory)
        
        Returns:
            List of matching file paths
        """
        try:
            cmd = f"find {directory} -name '{filename}'"
            output = self.env.execute(cmd)
            return output['output']
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error finding files: {str(e)}"
    
    def search_in_directory(self, pattern: str, directory: str = ".", use_regex: bool = True) -> str:
        """
        Search for a pattern in all files in a directory recursively.
        
        Args:
            pattern (str): pattern to search for
            directory (str): directory to search in (default: current directory)
            use_regex (bool): if False, treat the pattern as a fixed string
        
        Returns:
            Matching lines with file names and line numbers
        """
        try:
            esc = pattern.replace("'", "'\\''")
            fixed_flag = "-F" if not use_regex else ""
            cmd = f"grep {fixed_flag} -rn '{esc}' {directory}"
            output = self.env.execute(cmd)
            return output['output']
        except Exception as e:
            # grep returns non-zero when no matches found
            return f"No matches found or error: {str(e)}"
    
    # def get_file_content(self, file_path: str) -> str:
    #     """
    #     Return the entire content of the file as a string.
    #     """
    #     try:
    #         output = self.env.execute(f"cat {file_path}")
    #         return output['output']
    #     except Exception as e:
    #         return f"Error reading file: {str(e)}"

    def set_file_content(self, file_path: str, content: str) -> str:
        """
        Overwrite the file with the given content.
        """
        try:
            import tempfile
            import os
            
            # Create temp file on host system
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as host_temp:
                host_temp.write(content)
                host_temp_path = host_temp.name
            
            container_temp_path = '/tmp/swe_agent_temp_write.txt'
            
            try:
                # Use stdin piping to transfer file into container
                container_id = getattr(self.env, 'container_id', None)
                if not container_id:
                    raise ValueError("Container ID not available - this method requires DockerEnvironment")
                
                docker_executable = getattr(self.env.config, 'executable', 'docker')
                
                # Use docker exec with stdin to pipe content into container
                with open(host_temp_path, 'rb') as f:
                    file_content = f.read()
                
                # Use docker exec with stdin to write the file
                write_cmd = [
                    docker_executable, 'exec', '-i', container_id,
                    'bash', '-c', f'cat > {container_temp_path}'
                ]
                result = subprocess.run(
                    write_cmd,
                    input=file_content,
                    capture_output=True,
                    timeout=300  # 5 minute timeout for large files
                )
                
                if result.returncode != 0:
                    file_size = os.path.getsize(host_temp_path)
                    error_msg = f"Failed to write file to container. File size: {file_size} bytes. "
                    if result.stderr:
                        error_msg += f"Error: {result.stderr.decode('utf-8', errors='replace')}"
                    if result.stdout:
                        error_msg += f" Output: {result.stdout.decode('utf-8', errors='replace')}"
                    raise RuntimeError(error_msg)
                
                # Move temp file to target location in container
                self.env.execute(f"cp {container_temp_path} {file_path}")
                self.env.execute(f"rm -f {container_temp_path}")
                
                msg = f"Successfully wrote to {file_path}"
                return self._append_syntax_warning_if_needed(file_path, msg)
            
            finally:
                # Always clean up host temp file
                try:
                    os.unlink(host_temp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in set_file_content: {str(e)}"

    def regex_replace_in_file(self, file_path: str, pattern: str, replacement: str, use_regex: bool = True) -> str:
        """
        Replace pattern with replacement in the file (regex or literal).
        """
        try:
            esc_pattern = pattern.replace("'", "'\\''")
            esc_repl = replacement.replace("'", "'\\''")
            flag = "-E" if use_regex else ""
            cmd = f"sed -i {flag} 's/{esc_pattern}/{esc_repl}/g' {file_path}"
            self.env.execute(cmd)
            msg = f"Successfully replaced in {file_path}"
            return self._append_syntax_warning_if_needed(file_path, msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def replace_between(self, file_path: str, start_pattern: str, end_pattern: str, content: str,
                        use_regex: bool = False, include_start: bool = False, include_end: bool = False) -> str:
        """
        Replace the text between the first match of start_pattern and the first match of end_pattern.
        Safer than line-number editing when ranges shift. Patterns can be treated as fixed strings by default.

        Args:
            file_path: File to edit
            start_pattern: Anchor marking the start of the region
            end_pattern: Anchor marking the end of the region (searched after start)
            content: Replacement text for the region
            use_regex: If True, treat patterns as extended regex; otherwise fixed strings
            include_start: If True, the start anchor is also replaced
            include_end: If True, the end anchor is also replaced

        Returns:
            Summary string describing the change, or error message
        """
        try:
            # Read file
            read_output_d = self.env.execute(f"cat {file_path}")
            text = read_output_d['output']
            lines = text.split('\n')

            import re
            def find_index(pattern: str, start: int = 0) -> int:
                if use_regex:
                    try:
                        compiled_pattern = re.compile(pattern)
                    except re.error as e:
                        raise ValueError(
                            f"Invalid regex pattern: {e}. "
                            f"If you want literal string matching, set use_regex=False (the default). "
                            f"If you want regex, escape special characters properly."
                        )
                    for idx in range(start, len(lines)):
                        if compiled_pattern.search(lines[idx]):
                            return idx
                    return -1
                else:
                    return next((i for i in range(start, len(lines)) if pattern in lines[i]), -1)

            start_idx = find_index(start_pattern, 0)
            if start_idx == -1:
                return f"Error: start_pattern not found in {file_path}"
            end_idx = find_index(end_pattern, start_idx + 1)
            if end_idx == -1:
                return f"Error: end_pattern not found in {file_path}"

            replace_from = start_idx if include_start else start_idx + 1
            replace_to = end_idx if include_end else end_idx
            if replace_from > replace_to:
                return "Error: Computed replace range is invalid (check include_start/include_end)."

            # Preserve indentation of the line at replace_from (best-effort)
            indent = 0
            if 0 <= replace_from < len(lines):
                ref = lines[replace_from]
                indent = len(ref) - len(ref.lstrip(' \t'))
            adjusted = '\n'.join(((' ' * indent) + l if l else l) for l in content.split('\n'))

            new_lines = lines[:replace_from] + adjusted.split('\n') + lines[replace_to:]
            new_text = '\n'.join(new_lines)

            msg = self.set_file_content(file_path, new_text)
            # set_file_content already appends syntax warnings if needed
            return msg
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in replace_between: {str(e)}"

    def insert_lines_at(self, file_path: str, line_num: int, content: str, match_indentation: bool = True) -> str:
        """
        Insert content at the given line number (1-indexed), optionally matching surrounding indentation.
        """
        try:
            # Validate and convert line_num to int
            try:
                line_num = int(line_num)
            except (ValueError, TypeError):
                return f"Error: Invalid line_num. line_num={line_num}. It must be a valid integer."
            
            if line_num < 1:
                return f"Error: line_num must be >= 1, got {line_num}"
            
            # Read the file
            read_output_d = self.env.execute(f"cat {file_path}")
            file_content = read_output_d['output']
            lines = file_content.split('\n')
            
            if match_indentation and line_num > 1:
                # Get prev line to detect indent
                prev = lines[line_num - 2] if line_num - 2 < len(lines) else ""
                indent = len(prev) - len(prev.lstrip())
                content = '\n'.join(' ' * indent + l for l in content.split('\n'))
            
            # Insert content at specified line
            content_lines = content.split('\n')
            new_lines = lines[:line_num-1] + content_lines + lines[line_num-1:]
            new_content = '\n'.join(new_lines)
            
            # Write back using set_file_content (which handles large files)
            result = self.set_file_content(file_path, new_content)
            if result.startswith("Error"):
                return result
            
            msg = f"Successfully inserted content at line {line_num} in {file_path}"
            return self._append_syntax_warning_if_needed(file_path, msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in insert_lines_at: {str(e)}"

    def delete_lines(self, file_path: str, from_line: int, to_line: int) -> str:
        """
        Delete lines from from_line to to_line (1-indexed, inclusive).
        """
        try:
            # Validate and convert line numbers to int
            try:
                from_line = int(from_line)
            except (ValueError, TypeError):
                return f"Error: Invalid from_line. from_line={from_line}. It must be a valid integer."
            try:
                to_line = int(to_line)
            except (ValueError, TypeError):
                return f"Error: Invalid to_line. to_line={to_line}. It must be a valid integer."
            
            if from_line < 1 or to_line < 1 or from_line > to_line:
                return f"Error: Invalid line numbers. from_line={from_line}, to_line={to_line}"
            
            cmd = f"sed -i '{from_line},{to_line}d' {file_path}"
            self.env.execute(cmd)
            msg = f"Successfully deleted lines {from_line}-{to_line} in {file_path}"
            return self._append_syntax_warning_if_needed(file_path, msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def run_python_snippet(self, code: str) -> str:
        """
        Run the given Python code in the container and return output.
        Useful for testing or complex file operations.
        """
        try:
            import tempfile
            import os
            
            # Create temp file on host system
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as host_temp:
                host_temp.write(code)
                host_temp_path = host_temp.name
            
            container_temp_path = '/tmp/swe_agent_temp_snippet.py'
            
            try:
                # Use stdin piping to transfer file into container
                container_id = getattr(self.env, 'container_id', None)
                if not container_id:
                    raise ValueError("Container ID not available - this method requires DockerEnvironment")
                
                docker_executable = getattr(self.env.config, 'executable', 'docker')
                
                # Use docker exec with stdin to pipe content into container
                with open(host_temp_path, 'rb') as f:
                    file_content = f.read()
                
                # Use docker exec with stdin to write the file
                write_cmd = [
                    docker_executable, 'exec', '-i', container_id,
                    'bash', '-c', f'cat > {container_temp_path}'
                ]
                result = subprocess.run(
                    write_cmd,
                    input=file_content,
                    capture_output=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    error_msg = "Failed to write Python snippet to container. "
                    if result.stderr:
                        error_msg += f"Error: {result.stderr.decode('utf-8', errors='replace')}"
                    raise RuntimeError(error_msg)
                
                # Execute the Python file and clean up
                output = self.env.execute(f"python3 {container_temp_path}")
                self.env.execute(f"rm -f {container_temp_path}")
                return output['output']
            
            finally:
                # Always clean up host temp file
                try:
                    os.unlink(host_temp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in run_python_snippet: {str(e)}"
    
    def run_tests(self, test_cmd: "str | None" = None) -> str:
        """
        Run the test suite or specific tests to validate changes.
        If no test_cmd provided, tries to auto-detect test command.
        
        Args:
            test_cmd (str, optional): Test command to run (e.g., "pytest tests/test_file.py")
        
        Returns:
            Test output
        """
        try:
            if test_cmd is None:
                # Try to auto-detect test framework
                for cmd in ["pytest --co -q", "python -m pytest --co -q", "python manage.py test --help", "python -m unittest --help"]:
                    try:
                        result = self.env.execute(cmd)
                        if "pytest" in cmd and result.get('output'):
                            test_cmd = "pytest -xvs"
                            break
                        elif "manage.py" in cmd:
                            test_cmd = "python manage.py test"
                            break
                        elif "unittest" in cmd:
                            test_cmd = "python -m unittest discover"
                            break
                    except:
                        continue
                
                if test_cmd is None:
                    return "Could not auto-detect test command. Please provide test_cmd argument."
            
            result = self.env.execute(f"timeout 120 {test_cmd}")
            return result['output']
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Test execution error: {str(e)}"

    def detect_indentation(self, file_path: str) -> str:
        """
        Return indentation info (e.g., '4 spaces' or 'tabs').
        """
        try:
            # Read file content
            result = self.env.execute(f"cat {file_path}")
            content = result['output']
            for line in content.split('\n'):
                if line.startswith(' '):
                    return f"{len(line) - len(line.lstrip())} spaces"
                elif line.startswith('\t'):
                    return "tabs"
            return "No indentation detected"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def git_diff(self) -> str:
        """
        Show current git diff to see what changes have been made so far.
        CRITICAL: Call this before finish() to verify you made actual code changes.
        If this returns "No changes yet", you haven't modified any files and should NOT call finish()!
        
        Returns:
            Git diff output showing all current changes
        """
        try:
            result = self.env.execute("git diff")
            diff_output = result['output'].strip() if result['output'] else ""
            if not diff_output:
                return "No changes yet. You have not modified any files. Make code changes before calling finish()!"
            return diff_output
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error getting git diff: {str(e)}"
    
    def find_and_replace_text(self, file_path: str, old_text: str, new_text: str, count: int = 1) -> str:
        """
        Find and replace exact text in a file without using line numbers.
        Useful for small, precise changes where you know the exact text to replace.
        
        Advantages:
        - No line numbers needed (avoids stale line number issues)
        - Automatically preserves surrounding code
        - Good for small, targeted changes
        - Less prone to indentation errors
        
        When to use this tool:
        - Changing a single function call or variable name
        - Modifying a small code block (< 10 lines)
        - Making precise changes where you know exact text
        - Text to replace is unique in the file
        
        When to use replace_in_file instead:
        - Large sections (>20 lines) need rewriting
        - Text is not unique enough in file
        - Adding completely new code blocks
        - Multiple scattered edits across the file
        
        Args:
            file_path (str): path to the file to edit
            old_text (str): exact text to find and replace (must match exactly, including whitespace)
            new_text (str): text to replace it with
            count (int): maximum number of occurrences to replace (default: 1, use -1 for all)
        
        Returns:
            Success message or error if text not found
        """
        try:
            # Validate and convert count to int
            try:
                count = int(count)
            except (ValueError, TypeError):
                return f"Error: Invalid count. count={count}. It must be a valid integer."
            
            # Read the file
            read_output_d = self.env.execute(f"cat {file_path}")
            file_content = read_output_d['output']
            
            # Count occurrences
            occurrences = file_content.count(old_text)
            if occurrences == 0:
                return f"Error: Text not found in {file_path}. Make sure old_text matches EXACTLY (including all whitespace, indentation, and newlines)."
            
            # Perform replacement
            if count == -1:
                new_content = file_content.replace(old_text, new_text)
                replaced_count = occurrences
            else:
                new_content = file_content.replace(old_text, new_text, count)
                replaced_count = min(count, occurrences)
            
            # Write the file back using set_file_content (which handles encoding properly)
            write_result = self.set_file_content(file_path, new_content)
            
            msg = (
                f"✓ Successfully replaced {replaced_count} occurrence(s) in {file_path}.\n"
                f"Found {occurrences} total occurrence(s) of the text.\n"
                f"TIP: Call show_file to verify the change was correct."
            )
            
            # For Python files, append syntax check
            return self._append_syntax_warning_if_needed(file_path, msg)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in find_and_replace_text: {str(e)}"
    
    def check_syntax(self, file_path: str) -> str:
        """
        Check if a Python file has valid syntax after editing.
        Useful to quickly verify you didn't introduce syntax errors.
        
        Args:
            file_path (str): path to the Python file to check
            
        Returns:
            Success message or syntax error details
        """
        try:
            # Try to compile the file to check syntax
            # Note: execute() returns {"output": ..., "returncode": ...} and does NOT raise on non-zero exit
            result = self.env.execute(f"python3 -m py_compile {file_path} 2>&1")
            output = result['output']
            returncode = result.get('returncode', 0)
            
            if returncode == 0:
                # Command succeeded - no syntax errors
                return f"✓ {file_path} has valid Python syntax"
            else:
                # Non-zero exit code means syntax error
                return f"✗ Syntax error in {file_path}:\n{output.strip()}"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            # If an exception occurred (timeout, etc.), try alternative check
            
            # Try alternative check with ast.parse as fallback
            try:
                result = self.env.execute(f"python3 -c 'import ast; ast.parse(open(\"{file_path}\").read())' 2>&1")
                output = result['output']
                returncode = result.get('returncode', 0)
                
                if returncode == 0:
                    return f"✓ {file_path} has valid Python syntax"
                else:
                    return f"✗ Syntax error in {file_path}:\n{output.strip()}"
            except Exception as e2:
                import traceback
                traceback.print_exc()
                # Both methods failed - return error
                return f"✗ Error checking syntax for {file_path}: {str(e2)}"

    def list_modified_python_files(self) -> str:
        """
        List modified (unstaged) Python files according to git.
        """
        try:
            out = self.env.execute("git diff --name-only -- '*.py'")
            return out['output'] or ""
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error listing modified files: {str(e)}"

    def check_repo_syntax(self) -> str:
        """
        Check syntax for all modified Python files in the repository.
        """
        try:
            files_out = self.env.execute("git diff --name-only -- '*.py'")
            files = [f.strip() for f in files_out['output'].split('\n') if f.strip()]
            # if not files:
            #     files_out = self.env.execute("git ls-files -- '*.py'")
            #     files = [f.strip() for f in files_out['output'].split('\n') if f.strip()]
            if not files:
                return "No Python files to check."
            errors = []
            for f in files:
                # Use check_syntax which properly handles exceptions
                syntax_result = self.check_syntax(f)
                if syntax_result.strip().startswith('✗') or 'Syntax error' in syntax_result:
                    errors.append(f"{f}:\n{syntax_result}")
            if errors:
                return "✗ Syntax errors detected:\n\n" + "\n\n".join(errors)
            return "✓ All checked Python files have valid syntax"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during repository syntax check: {str(e)}"

    def git_apply(self, patch: str) -> str:
        """
        Apply a unified diff patch string using git apply.

        Returns success or stderr on failure.
        """
        try:
            import tempfile
            import os
            
            # Create temp file on host system
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.patch', encoding='utf-8') as host_temp:
                host_temp.write(patch)
                host_temp_path = host_temp.name
            
            container_temp_path = '/tmp/swe_agent_temp_patch.patch'
            
            try:
                # Use stdin piping to transfer file into container
                container_id = getattr(self.env, 'container_id', None)
                if not container_id:
                    raise ValueError("Container ID not available - this method requires DockerEnvironment")
                
                docker_executable = getattr(self.env.config, 'executable', 'docker')
                
                # Use docker exec with stdin to pipe content into container
                with open(host_temp_path, 'rb') as f:
                    file_content = f.read()
                
                # Use docker exec with stdin to write the file
                write_cmd = [
                    docker_executable, 'exec', '-i', container_id,
                    'bash', '-c', f'cat > {container_temp_path}'
                ]
                result = subprocess.run(
                    write_cmd,
                    input=file_content,
                    capture_output=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    error_msg = "Failed to write patch to container. "
                    if result.stderr:
                        error_msg += f"Error: {result.stderr.decode('utf-8', errors='replace')}"
                    raise RuntimeError(error_msg)
                
                # Apply the patch and clean up
                apply_result = self.env.execute(f"git apply -p0 -v {container_temp_path}")
                self.env.execute(f"rm -f {container_temp_path}")
                
                if apply_result.get('exit_code', 0) == 0:
                    # After applying a patch, check syntax of modified Python files
                    try:
                        files_out = self.env.execute("git diff --name-only -- '*.py'")
                        files = [f.strip() for f in files_out.get('output', '').split('\n') if f.strip()]
                    except Exception:
                        files = []
                    warnings = []
                    for f in files:
                        try:
                            syntax = self.check_syntax(f)
                            if syntax.strip().startswith('✗') or 'Syntax error' in syntax:
                                warnings.append(f"{f}:\n{syntax}")
                        except Exception:
                            import traceback
                            traceback.print_exc()
                            continue
                    if warnings:
                        return "Patch applied successfully\n\nWarnings (syntax errors detected):\n\n" + "\n\n".join(warnings)
                    return "Patch applied successfully"
                return apply_result.get('output', 'git apply returned a non-zero exit code')
            
            finally:
                # Always clean up host temp file
                try:
                    os.unlink(host_temp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in git_apply: {str(e)}"

    def count_lines(self, file_path: str) -> str:
        """
        Return the total line count of a file.
        Useful to know the size before doing show_file() or to understand file structure.
        
        Args:
            file_path (str): path to the file
            
        Returns:
            Line count information or error message
        """
        try:
            result = self.env.execute(f"wc -l {file_path}")
            return result['output'].strip()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error counting lines: {str(e)}"
    
    def verify_before_finish(self) -> str:
        """
        Comprehensive pre-finish verification checklist. Call this BEFORE finish() to ensure quality.
        
        This function automatically:
        1. Shows git diff of all changes
        2. Checks Python syntax on modified files
        3. Verifies that actual code changes were made
        4. Returns a summary report with pass/fail status
        
        **IMPORTANT**: This function will tell you if it's safe to call finish() or if you need to fix issues first.
        
        Returns:
            Verification report with ✅ if passed or ❌ with specific issues to fix
        """
        try:
            report = []
            all_checks_passed = True
            
            # 1. Git diff check - verify changes were made
            report.append("=" * 60)
            report.append("PRE-FINISH VERIFICATION CHECKLIST")
            report.append("=" * 60)
            report.append("")
            
            diff = self.git_diff()
            if not diff or diff.strip() == "" or "No changes yet" in diff:
                report.append("❌ CHECK 1 FAILED: No code changes detected")
                report.append("   You have not modified any files.")
                report.append("   DO NOT call finish() - make code changes first!")
                return "\n".join(report)

            # Count lines changed
            diff_lines = diff.split('\n')
            additions = len([l for l in diff_lines if l.startswith('+') and not l.startswith('+++')])
            deletions = len([l for l in diff_lines if l.startswith('-') and not l.startswith('---')])

            report.append(f"✅ CHECK 1 PASSED: Code changes detected")
            report.append(f"   Lines added: {additions}")
            report.append(f"   Lines deleted: {deletions}")
            report.append("")
            
            # 2. Syntax check - verify no syntax errors (excluding test error files)
            syntax_result = self.check_repo_syntax()
            
            if "✗" in syntax_result or "Syntax errors detected" in syntax_result:
                report.append("❌ CHECK 2 FAILED: Syntax errors detected")
                report.append(f"   {syntax_result}")
                report.append("")
                report.append("   FIX THESE ERRORS before calling finish()!")
                all_checks_passed = False
            else:
                report.append(f"✅ CHECK 2 PASSED: {syntax_result}")
                report.append("")
            
            # 3. Show diff preview
            report.append("=" * 60)
            report.append("GIT DIFF PREVIEW (first 1000 chars):")
            report.append("=" * 60)
            diff_preview = diff[:1000]
            if len(diff) > 1000:
                diff_preview += "\n... (truncated, use git_diff() to see full diff)"
            report.append(diff_preview)
            report.append("")
            
            # 4. Final verdict
            report.append("=" * 60)
            if all_checks_passed:
                report.append("✅ ALL CHECKS PASSED - Safe to call finish()")
                report.append("=" * 60)
                report.append("")
                report.append("Final reminders before calling finish():")
                report.append("  • Review the diff above - does it match the task?")
                report.append("  • Are you modifying the RIGHT files (not test files)?")
                report.append("  • Is indentation correct (no misaligned code)?")
                report.append("  • Did you preserve imports and critical code?")
                report.append("")
                report.append("If all looks good, call: finish('your summary here')")
            else:
                report.append("❌ VERIFICATION FAILED - DO NOT call finish() yet")
                report.append("=" * 60)
                report.append("")
                report.append("Fix the issues above, then run verify_before_finish() again.")
            
            return "\n".join(report)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during verification: {str(e)}"

    def _append_syntax_warning_if_needed(self, file_path: str, base_message: str) -> str:
        """
        Append a small warning with syntax errors for Python files, if any.

        This runs a quick syntax check for `.py` files and, when errors are
        detected (including IndentationError, SyntaxError, etc.), appends a 
        concise warning that includes the actual error.
        """
        try:
            if isinstance(file_path, str) and file_path.endswith('.py'):
                syntax = self.check_syntax(file_path)
                if isinstance(syntax, str) and (syntax.strip().startswith('✗') or 'Syntax error' in syntax):
                    return f"{base_message}\n\nWarning: syntax errors detected in {file_path}:\n{syntax}"
        except Exception:
            import traceback
            traceback.print_exc()
            # Do not block main operation on warning generation
            pass
        return base_message

class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """
    
    def __init__(self):
        pass

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output
    
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        return self.execute(command)