"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history tree (role, content, timestamp, unique_id, parent, children)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`, and `add_instructions_and_backtrack`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

import os
from pathlib import Path
from typing import List, Callable, Dict, Any
import time

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect

SYSTEM_PROMPT = """You are a highly capable ReAct-style coding agent and expert software engineer. Your goal is to quickly diagnose coding tasks, make minimal, correct code edits, verify them, and finish with a clear summary.

Task objectives:
- Understand the issue and locate relevant code efficiently
- Make small, targeted edits with careful attention to indentation and line numbers
- Verify changes and finish decisively with a concise result

Response format (mandatory):
- Each response must include a brief reasoning (1–3 sentences) followed by exactly one function call block
- Use the exact function call block format below (markers must match exactly)
- Do not include any text after the END marker
- One function call per response only. It is mandatory to call exactly one function in your response.

Your response format (exact):
Brief reasoning here (1–3 sentences)
----BEGIN_FUNCTION_CALL----
function_name
----ARG----
arg1_name
arg1_value
----ARG----
arg2_name
arg2_value
----END_FUNCTION_CALL----

Format rules:
- Always include the `----BEGIN_FUNCTION_CALL----` marker and end with the `----END_FUNCTION_CALL----` marker as the last characters in the response
- Function name appears on its own line after the BEGIN marker
- Each argument consists of:
  - A line with ----ARG----
  - A line with the argument name
  - the argument value starting from the next line, until the next `----ARG----` or `----END_FUNCTION_CALL----` marker
- No text after ----END_FUNCTION_CALL----
- Exactly one function call per response

Recommended workflow:
1) Explore (focused)
   - Use search_in_directory to find relevant files or patterns
   - Use find_file to locate files by name
   - Use search_in_file for precise in-file queries
2) Read
   - Use show_file to inspect only the necessary lines (prefer ranges)
   - Confirm indentation style (tabs vs spaces) and surrounding context
3) Edit
   - For small, precise text changes: Use find_and_replace_text (safer, no line numbers)
   - For line-based edits: Use replace_in_file (requires exact line numbers)
   - Immediately re-read with show_file after every edit to refresh line numbers and verify the change
   - Repeat for additional edits (always re-read after each edit)
4) Test/Verify
   - Run tests or quick checks (e.g., run_bash_cmd, check_syntax for Python) as appropriate
5) Finish
   - Call git_diff to confirm actual changes
   - If the diff is correct, finish() with a brief summary of the fix

Key rules for safe editing:
- Always read the file before editing to understand current structure
- CRITICAL: After EVERY replace_in_file call, immediately call show_file on the edited section
  - Line numbers change after edits - using stale line numbers will corrupt files
  - This re-reading step is MANDATORY, not optional
  - Verify the edit was applied correctly before proceeding

INDENTATION RULES (CRITICAL FOR SUCCESS):
- BEFORE editing: Call detect_indentation(file_path) to see if file uses tabs or spaces
- Read the exact lines you'll replace - note the indentation level precisely
- Match indentation EXACTLY - count spaces/tabs character-by-character
- For Python files: After EVERY edit, call check_syntax(file_path) to catch indentation errors
- If syntax check fails, immediately fix and re-check before proceeding
- Common mistake: Copying indentation from system prompt examples instead of from the actual file

EDIT SIZE AND SAFETY:
- Maximum recommended edit: 20 lines per replace_in_file call
- For larger changes: Break into multiple small, sequential edits
- After each small edit: Re-read, verify, then proceed to next edit
- For massive refactorings: Use run_bash_cmd with sed/awk/python scripts instead
- Line numbers are 1-indexed and inclusive (from_line and to_line both included)
- from_line and to_line must be integers
- Make minimal changes; avoid unnecessary refactors
- Make decisions autonomously; do not ask the user for choices or input

Efficiency tips:
- Aim for 5–15 steps for most tasks
- Be concise and act quickly
- If the same approach fails repeatedly, try a different angle (e.g., a different file or method)
- Finish as soon as the fix is applied and verified

Common pitfalls to avoid (LEARN FROM THESE):
- Missing or malformed function call markers
- Text after ----END_FUNCTION_CALL----
- Multiple function calls in one response
- CRITICAL: Stale line numbers (not re-reading after edits) - causes 40% of failures
- CRITICAL: Indentation mismatches (tabs vs spaces) - causes 50% of failures
- Replacing too many lines at once (>20 lines) - hard to get indentation right
- Deleting imports or critical code unintentionally
- Creating duplicate functions/methods
- Finishing without making actual changes
- Finishing without calling git_diff to verify changes
- Asking the user for input or choices
- Not calling check_syntax after editing Python files

Search strategies:
- Start broad with search_in_directory; narrow with search_in_file
- Use specific patterns (function/class names, error messages)
- Limit reading to relevant line ranges with show_file

Bash best practices:
- Use run_bash_cmd to run tests or for larger scripted edits
- Prefer replace_in_file for small, precise changes
- For big edits, write a short script within run_bash_cmd rather than passing extremely large content to replace_in_file

How to finish (MANDATORY CHECKLIST):
Before calling finish(), complete ALL of these steps:
1. Run git_diff() and carefully review the changes
2. For Python files: Run check_repo_syntax() to verify no syntax/indentation errors
3. Verify the diff matches the task requirements:
   - Check for correct indentation (no misaligned code)
   - Check for NO deleted imports or critical code
   - Check for NO duplicate functions or methods
   - Check that only relevant code was modified
4. If issues found in diff: Fix them immediately, don't call finish()
5. Only call finish() when the diff is clean and correct
6. Do not finish if no changes were made or if changes are incorrect

Correct format examples:

Example 1 (search):
I need to find where authentication is handled.
----BEGIN_FUNCTION_CALL----
search_in_directory
----ARG----
pattern
def authenticate
----ARG----
directory
.
----END_FUNCTION_CALL----

Example 2 (read a range):
I’ll inspect the function to see current logic and indentation.
----BEGIN_FUNCTION_CALL----
show_file
----ARG----
file_path
src/auth.py
----ARG----
start_line
40
----ARG----
end_line
80
----END_FUNCTION_CALL----

Example workflow (concise):
1) search_in_directory → 2) show_file → 3) replace_in_file → 4) show_file (re-read) → 5) tests via run_bash_cmd → 6) git_diff → 7) finish

Work autonomously, keep edits minimal and precise, verify your work, and always end with a single correctly formatted function call block at every step."""

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history tree with unique ids
    - Builds the LLM context from the root to current node
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM, instance_id: str = 'default', output_dir: str = 'default'):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message tree storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1

        self.last_printed_message_id: int = -1

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Error tracking for recovery
        self.recent_errors: List[Dict[str, Any]] = []
        self.recovery_count: int = 0
        self.max_recoveries: int = 3  # Maximum number of times to attempt recovery
        
        # Set up the initial structure of the history
        # Create required root nodes and a user node (task) and an instruction node.
        system_prompt = SYSTEM_PROMPT
        self.system_message_id = self.add_message("system", system_prompt)
        self.user_message_id = self.add_message("user", "")
        self.instructions_message_id = self.add_message("instructor", "")

        self.instance_id = instance_id
        
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish, self.add_instructions_and_backtrack])

        self.output_dir = output_dir
        self.instance_id = instance_id

    # -------------------- MESSAGE TREE --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the tree.

        The message must include fields: role, content, timestamp, unique_id, parent, children.
        Maintain a pointer to the current node and the root node.
        """
        # Create new message
        unique_id = len(self.id_to_message)
        message = {
            "role": role,
            "content": content,
            "timestamp": int(time.time()),
            "unique_id": unique_id,
            "parent": self.current_message_id if self.current_message_id != -1 else None,
            "children": []
        }
        
        # Link to parent
        if self.current_message_id != -1:
            parent = self.id_to_message[self.current_message_id]
            parent["children"].append(unique_id)
        
        # Add to storage
        self.id_to_message.append(message)
        
        # Update pointers
        if self.root_message_id == -1:
            self.root_message_id = unique_id
        self.current_message_id = unique_id
        
        return unique_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """Update message content by id."""
        if message_id < 0 or message_id >= len(self.id_to_message):
            raise ValueError(f"Invalid message_id: {message_id}")
        self.id_to_message[message_id]["content"] = content

    def get_context(self) -> str:
        """
        Build the full LLM context by walking from the root to the current message.
        """
        # Build path from root to current
        path = []
        current_id = self.current_message_id
        
        while current_id is not None and current_id != -1:
            path.append(current_id)
            message = self.id_to_message[current_id]
            current_id = message["parent"]
        
        path.reverse()
        
        # Build context string
        context_parts = []
        for msg_id in path:
            context_parts.append(self.message_id_to_context(msg_id))
        
        return "".join(context_parts)

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        for tool in tools:
            self.function_map[tool.__name__] = tool

    def finish(self, result: str):
        """The agent must call this function with the final result when it has solved the given task. The function calls "git add -A and git diff --cached" to generate a patch and returns the patch as submission.

        Args: 
            result (str); the result generated by the agent

        Returns:
            The result passed as an argument.  The result is then returned by the agent's run method.
        """
        return result 

    def add_instructions_and_backtrack(self, instructions: str, at_message_id: int):
        """
        The agent should call this function if it is making too many mistakes or is stuck.

        The function changes the content of the instruction node with 'instructions' and
        backtracks at the node with id 'at_message_id'. Backtracking means the current node
        pointer moves to the specified node and subsequent context is rebuilt from there.

        Returns a short success string.
        """
        # Update instruction node content
        self.set_message_content(self.instructions_message_id, instructions)

        # Backtrack to specified message
        if at_message_id < 0 or at_message_id >= len(self.id_to_message):
            raise ValueError(f"Invalid at_message_id: {at_message_id}")

        self.current_message_id = at_message_id

        return f"Instructions updated and backtracked to message {at_message_id}."
    
    # -------------------- ERROR DETECTION AND RECOVERY --------------------
    def detect_error_loop(self, error_type: str, error_msg: str) -> bool:
        """
        Detect if the agent is stuck in an error loop.
        
        Returns True if the agent should be recovered (same error repeated multiple times).
        """
        # Add current error to recent errors
        self.recent_errors.append({
            "type": error_type,
            "message": error_msg,
            "step": len(self.id_to_message)
        })
        
        # Keep only last 10 errors
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)
        
        # Check for error loops: same error type appearing 3+ times in last 5 errors
        if len(self.recent_errors) >= 5:
            recent_5 = self.recent_errors[-5:]
            error_counts = {}
            for err in recent_5:
                error_counts[err["type"]] = error_counts.get(err["type"], 0) + 1
            
            # If any error type appears 3+ times in last 5 errors, we're stuck
            if any(count >= 3 for count in error_counts.values()):
                return True
        
        return False
    
    def perform_recovery(self, step: int) -> bool:
        """
        Perform recovery by backtracking and adding helpful instructions.
        
        Returns True if recovery was performed, False if max recoveries exceeded.
        """
        # Check if we've exceeded max recoveries
        if self.recovery_count >= self.max_recoveries:
            return False
        
        self.recovery_count += 1
        
        # Find a good point to backtrack to (before the error loop started)
        # Go back to the last successful tool call (not an error message)
        backtrack_target = self.current_message_id
        steps_back = 0
        max_steps_back = 10  # Don't go back more than 10 messages
        
        while steps_back < max_steps_back and backtrack_target > self.instructions_message_id:
            msg = self.id_to_message[backtrack_target]
            # Find a tool message that's not an error
            if msg["role"] == "tool" and not msg["content"].startswith("Error"):
                break
            backtrack_target = msg.get("parent", self.instructions_message_id)
            steps_back += 1
        
        # If we couldn't find a good point, go back to instructions
        if backtrack_target <= self.instructions_message_id:
            backtrack_target = self.instructions_message_id
        
        # Analyze the error pattern to provide specific guidance
        error_types = [err["type"] for err in self.recent_errors[-5:]]
        
        recovery_instructions = ""
        if "parsing" in str(error_types).lower():
            recovery_instructions = """
⚠️ RECOVERY MODE: You've been stuck in parsing errors. 

CRITICAL FIX:
1. Every response MUST end with EXACTLY this format:
   ----BEGIN_FUNCTION_CALL----
   function_name
   ----ARG----
   arg_name
   arg_value
   ----END_FUNCTION_CALL----

2. NO text after ----END_FUNCTION_CALL----
3. NO duplicate markers
4. Write 1 sentence reasoning, then immediately call a function

Try a simpler action now. If you were reading files, try making an edit. If you were editing, verify with git_diff."""
        elif "function" in str(error_types).lower() or "unexpected keyword" in str(error_types).lower():
            recovery_instructions = """
⚠️ RECOVERY MODE: You've been calling functions incorrectly.

CRITICAL FIX:
1. Check the function signature carefully
2. Only pass arguments that the function accepts
3. Use simple, straightforward function calls
4. Common functions:
   - search_in_directory(pattern, directory)
   - show_file(file_path, start_line, end_line)
   - replace_in_file(file_path, from_line, to_line, content)
   - git_diff()
   - finish(result)

Try a different action now. Simplify your approach."""
        else:
            recovery_instructions = f"""
⚠️ RECOVERY MODE: You've been stuck in repeated errors (recovery attempt {self.recovery_count}/{self.max_recoveries}).

CRITICAL FIX:
1. Stop repeating the same approach
2. Try a COMPLETELY different action
3. If stuck on editing, try this sequence:
   a) Read the file section you want to edit
   b) Note the EXACT indentation (count spaces/tabs)
   c) Make a SMALL edit (max 10-15 lines)
   d) Immediately re-read to verify
   e) If syntax error, fix immediately
4. Common mistakes to avoid:
   - Using stale line numbers (re-read after each edit!)
   - Wrong indentation (match existing code exactly!)
   - Replacing too many lines at once (break into smaller edits!)
   - Deleting imports or critical code

Recent error pattern: {error_types[-3:]}

Take a different approach now. Make a simple, safe action."""
        
        # Add recovery message to the tree
        self.current_message_id = backtrack_target
        self.add_message("instructor", recovery_instructions)
        
        # Clear recent errors since we're recovering
        self.recent_errors = []
        
        print(f"🔄 Recovery performed: backtracked to message {backtrack_target} (recovery {self.recovery_count}/{self.max_recoveries})")
        
        return True

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message tree
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the tree
            - If `finish` is called, return the final result
        """
        # Ensure max_steps is capped at 100
        max_steps = min(max_steps, 100)

        # Set the user prompt
        self.set_message_content(self.user_message_id, task)

        # Main ReAct loop
        for step in range(max_steps):
            if self.current_message_id != self.last_printed_message_id:
                # Print messages from last printed message id to current message id
                for msg_id in range(self.last_printed_message_id + 1, self.current_message_id + 1):
                    # print(self.message_id_to_context(msg_id))
                    # Write to file Path(self.output_dir) / "exec_trajectories" / f"{self.instance_id}.txt"
                    os.makedirs(Path(self.output_dir) / "exec_trajectories", exist_ok=True)
                    with open(Path(self.output_dir) / "exec_trajectories" / f"{self.instance_id}.txt", "a") as f:
                        f.write(self.message_id_to_context(msg_id))
                        f.write("\n")

                self.last_printed_message_id = self.current_message_id

            try:
                # Build context from message tree
                context = self.get_context()
                
                # Query the LLM
                response = self.llm.generate(context)
                
                # Add assistant response to tree
                self.add_message("assistant", response)
                
                # Parse the function call
                try:
                    parsed = self.parser.parse(response)
                    function_name = parsed["name"]
                    arguments = parsed["arguments"]
                    
                    # Check if function exists
                    if function_name not in self.function_map:
                        error_msg = f"Error: Function '{function_name}' not found. Available functions: {list(self.function_map.keys())}"
                        self.add_message("tool", error_msg)
                        
                        # Detect and recover from error loops
                        if self.detect_error_loop("function_not_found", error_msg):
                            if self.perform_recovery(step):
                                continue
                        continue
                    
                    # Get the function
                    func = self.function_map[function_name]

                    # Sanitize arguments against function signature to avoid unexpected keyword errors
                    try:
                        sig = inspect.signature(func)
                        param_map = sig.parameters
                        # keep only accepted kwargs
                        filtered_args = {k: v for k, v in arguments.items() if k in param_map}
                        # detect missing required parameters
                        missing = []
                        for name, p in param_map.items():
                            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                                if p.default is inspect._empty and name not in filtered_args:
                                    missing.append(name)
                        if missing:
                            error_msg = f"Error: Missing required arguments for {function_name}: {missing}. Use the exact tool signature {func.__name__}{sig}."
                            self.add_message("tool", error_msg)
                            if self.detect_error_loop("function_call_missing_args", error_msg):
                                if self.perform_recovery(step):
                                    continue
                            continue
                        # Execute the tool
                        result = func(**filtered_args)
                        
                        # If finish was called, return the result
                        if function_name == "finish":
                            oo = self.function_map["generate_patch"](result)
                            if "diff --git" not in oo:
                                # The agent has not actually mande any code changes. Add error message
                                error_msg = "Error: finish() must be called only after making code changes. You must use the file edit tools to make changes to the codebase to resolve the issue. After making changes, you must call finish() to indicate that the task has been completed."
                                self.add_message("tool", error_msg)
                                continue

                            # Additional guard: ensure repository has no Python syntax errors before finishing
                            repo_syntax_checker = self.function_map["check_repo_syntax"]
                            try:
                                syntax_report = repo_syntax_checker()
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                syntax_report = None
                            # Expect formats from envs.SWEEnvironment.check_repo_syntax
                            if isinstance(syntax_report, str) and (syntax_report.strip().startswith("✗") or "Syntax errors detected" in syntax_report):
                                # Surface the filepaths and error messages to the transcript and continue
                                error_msg = (
                                    "Error: Repository has Python syntax errors. Fix these before calling finish().\n\n"
                                    + syntax_report
                                )
                                self.add_message("tool", error_msg)
                                continue

                            return result

                        # Add tool result to tree
                        self.add_message("tool", str(result))

                    except TypeError as e:
                        # Handle argument mismatch errors
                        import traceback
                        traceback.print_exc()
                        error_msg = f"Error calling {function_name}: {str(e)}"
                        self.add_message("tool", error_msg)
                        
                        # Detect and recover from error loops
                        if self.detect_error_loop("function_call_error", error_msg):
                            if self.perform_recovery(step):
                                continue
                    except Exception as e:
                        # Handle other execution errors
                        import traceback
                        traceback.print_exc()
                        error_msg = f"Error executing {function_name}: {str(e)}"
                        self.add_message("tool", error_msg)
                        
                        # Detect and recover from error loops
                        if self.detect_error_loop("execution_error", error_msg):
                            if self.perform_recovery(step):
                                continue
                
                except ValueError as e:
                    # Handle parsing errors
                    import traceback
                    traceback.print_exc()
                    error_msg = f"Error parsing function call: {str(e)}"
                    self.add_message("tool", error_msg)
                    
                    # Detect and recover from error loops
                    if self.detect_error_loop("parsing_error", error_msg):
                        if self.perform_recovery(step):
                            continue
            
            except Exception as e:
                # Handle unexpected errors in the loop
                import traceback
                traceback.print_exc()
                error_msg = f"Unexpected error in step {step}: {str(e)}"
                self.add_message("tool", error_msg)
                
                # Detect and recover from error loops
                if self.detect_error_loop("unexpected_error", error_msg):
                    if self.perform_recovery(step):
                        continue
        
        if self.current_message_id != self.last_printed_message_id:
            # Print messages from last printed message id to current message id
            for msg_id in range(self.last_printed_message_id + 1, self.current_message_id + 1):
                # print(self.message_id_to_context(msg_id))
                # Write to file Path(self.output_dir) / "exec_trajectories" / f"{self.instance_id}.txt"
                os.makedirs(Path(self.output_dir) / "exec_trajectories", exist_ok=True)
                with open(Path(self.output_dir) / "exec_trajectories" / f"{self.instance_id}.txt", "a") as f:
                    f.write(self.message_id_to_context(msg_id))
                    f.write("\n")

            self.last_printed_message_id = self.current_message_id
        
        # If we reach max_steps without finishing
        return f"Agent reached maximum steps ({max_steps}) without completing the task."

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        elif message["role"] == "instructor":
            return f"{header}YOU MUST FOLLOW THE FOLLOWING INSTRUCTIONS AT ANY COST. OTHERWISE, YOU WILL BE DECOMISSIONED.\n{content}\n"
        else:
            return f"{header}{content}\n"

def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-5-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=100)
    print(result)

if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()