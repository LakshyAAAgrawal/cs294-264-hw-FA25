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
        Line numbers are 1-indexed.
        
        Args:
            file_path (str): path to the file to edit
            from_line (int): starting line number (1-indexed, inclusive)
            to_line (int): ending line number (1-indexed, inclusive)
            content (str): new content to replace the lines
        
        Returns:
            Success message or error description
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
            new_lines = lines[:from_line-1] + [content] + lines[to_line:]
            new_content = '\n'.join(new_lines)
            
            # Write back to file using Python for reliability
            import base64
            encoded_content = base64.b64encode(new_content.encode()).decode()
            cmd = f"echo '{encoded_content}' | base64 -d > {file_path}"
            
            self.env.execute(cmd)
            return f"Successfully replaced lines {from_line}-{to_line} in {file_path}"
            
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
                cmd = f"sed -n '{start_line},{end_line}p' {file_path} | cat -n"
            elif start_line is not None:
                # Show from start_line to end
                cmd = f"sed -n '{start_line},$p' {file_path} | cat -n"
            else:
                # Show entire file
                cmd = f"cat -n {file_path}"
            
            output = self.env.execute(cmd)
            return output['output']
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def search_in_file(self, file_path: str, pattern: str) -> str:
        """
        Search for a pattern in a file and return matching lines with line numbers.
        
        Args:
            file_path (str): path to the file to search
            pattern (str): pattern to search for
        
        Returns:
            Matching lines with line numbers
        """
        try:
            cmd = f"grep -n '{pattern}' {file_path}"
            output = self.env.execute(cmd)
            return output['output']
        except Exception as e:
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
            return f"Error finding files: {str(e)}"
    
    def search_in_directory(self, pattern: str, directory: str = ".") -> str:
        """
        Search for a pattern in all files in a directory recursively.
        
        Args:
            pattern (str): pattern to search for
            directory (str): directory to search in (default: current directory)
        
        Returns:
            Matching lines with file names and line numbers
        """
        try:
            cmd = f"grep -rn '{pattern}' {directory}"
            output = self.env.execute(cmd)
            return output['output']
        except Exception as e:
            # grep returns non-zero when no matches found
            return f"No matches found or error: {str(e)}"
    
    def get_file_content(self, file_path: str) -> str:
        """
        Return the entire content of the file as a string.
        """
        try:
            output = self.env.execute(f"cat {file_path}")
            return output['output']
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def set_file_content(self, file_path: str, content: str) -> str:
        """
        Overwrite the file with the given content.
        """
        try:
            import base64
            encoded = base64.b64encode(content.encode()).decode()
            cmd = f"echo '{encoded}' | base64 -d > {file_path}"
            self.env.execute(cmd)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            # Fallback for large content: Write to temp and mv
            temp_path = "/tmp/temp_content"
            self.env.execute(f"echo '{encoded}' | base64 -d > {temp_path} && mv {temp_path} {file_path}")
            return f"Successfully wrote to {file_path}"

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
            return f"Successfully replaced in {file_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    def insert_lines_at(self, file_path: str, line_num: int, content: str, match_indentation: bool = True) -> str:
        """
        Insert content at the given line number (1-indexed), optionally matching surrounding indentation.
        """
        try:
            if match_indentation:
                # Get prev line to detect indent
                prev = self.env.execute(f"sed -n '{line_num-1}p' {file_path}")
                indent = len(prev) - len(prev.lstrip())
                content = '\n'.join(' ' * indent + l for l in content.split('\n'))
            import base64
            encoded = base64.b64encode(content.encode()).decode()
            cmd = f"sed -i '{line_num}i $(echo '{encoded}' | base64 -d)' {file_path}"
            self.env.execute(cmd)
            return "Success"
        except Exception as e:
            return f"Error: {str(e)}"

    def delete_lines(self, file_path: str, from_line: int, to_line: int) -> str:
        """
        Delete lines from from_line to to_line (1-indexed, inclusive).
        """
        try:
            cmd = f"sed -i '{from_line},{to_line}d' {file_path}"
            self.env.execute(cmd)
            return "Success"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_python_snippet(self, code: str) -> str:
        """
        Run the given Python code in the container and return output.
        """
        try:
            import base64
            encoded = base64.b64encode(code.encode()).decode()
            cmd = f"python3 -c \"$(echo '{encoded}' | base64 -d)\""
            return self.env.execute(cmd)['output']
        except Exception as e:
            return f"Error: {str(e)}"

    def detect_indentation(self, file_path: str) -> str:
        """
        Return indentation info (e.g., '4 spaces' or 'tabs').
        """
        try:
            content = self.get_file_content(file_path)
            for line in content.split('\n'):
                if line.startswith(' '):
                    return f"{len(line) - len(line.lstrip())} spaces"
                elif line.startswith('\t'):
                    return "tabs"
            return "No indentation detected"
        except Exception as e:
            return f"Error: {str(e)}"

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