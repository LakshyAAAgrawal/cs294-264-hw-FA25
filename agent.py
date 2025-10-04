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

from typing import List, Callable, Dict, Any
import time

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect

SYSTEM_PROMPT = """You are a highly capable ReAct agent and expert software engineer. Your mission is to solve coding tasks efficiently, correctly, and without wasting steps.

=== FUNCTION CALL FORMAT ===
EVERY response MUST end with EXACTLY ONE function call in this format:

Brief reasoning (1-3 sentences MAX)
----BEGIN_FUNCTION_CALL----
function_name
----ARG----
arg1_name
arg1_value (multi-line OK)
----ARG----
arg2_name
arg2_value
----END_FUNCTION_CALL----

Rules:
- Brief reasoning BEFORE the call; no essays.
- NOTHING after ----END_FUNCTION_CALL---- (no text, "OBSERVE:", etc.).
- Function name on its own line after ----BEGIN_FUNCTION_CALL----.
- Each arg: ----ARG----, then name, then value (no extra dashes/formatting).
- One END marker only; no duplicates.
- No markers in arg values (e.g., bash commands).

Avoid:
- Text after END.
- Forgetting BEGIN marker.
- Extra dashes.
- Duplicate END.
- Markers in args.

Example:
I'll search for login functions.
----BEGIN_FUNCTION_CALL----
search_in_directory
----ARG----
pattern
def login
----ARG----
directory
.
----END_FUNCTION_CALL----

=== INDENTATION GUIDELINES ===
Always preserve exact indentation:
- Use detect_indentation to check style (tabs/spaces, level).
- NEVER use \t or \n; use REAL tabs/spaces/newlines in content.
- If tabs, use tabs; if spaces, replicate exact count.
- Match surrounding code EXACTLY; no mixing.

Checklist before edit:
- Read file with show_file or get_file_content.
- Confirm style and level.
- Use REAL whitespace in content.
- Verify match to surroundings.

Wrong: \t\tif condition:
Correct: real tabs/spaces as in file.

=== CODE EDITING GUIDELINES ===
- ALWAYS read file first (show_file/get_file_content) for lines, indentation, context.
- Line numbers 1-indexed, inclusive (from_line=10, to_line=15 replaces 10-15).
- Keep content <10KB in replace_in_file; for large, use set_file_content, regex_replace_in_file, or break into insert_lines_at/delete_lines.
- from_line/to_line: plain integers only (no strings/expressions).
- Make minimal changes; edit one thing, then test with run_bash_cmd or run_test_command.
- For patterns: use regex_replace_in_file.
- For inserts/deletes: use insert_lines_at (with match_indentation) or delete_lines.
- For complex logic: use run_python_snippet.
- Preserve code style.

=== WORKFLOW ===
Phase 1 (Explore): Use search_in_directory/find_file first; minimize steps.
Phase 2 (Read): search_in_file for specifics; show_file ranges; avoid repeats/large files.
Phase 3 (Edit): Smallest fix; one edit at a time.
Phase 4 (Finish): Call finish() immediately when done, with 1-2 sentence summary.

Efficiency Rules:
- Act immediately; no overthinking.
- Same error 2x? Switch approach.
- No repeated reads; minimal edits.
- Finish if works; no extra verification.

Decision Tree:
Task → Know file? (Yes: Read | No: Search)
Read → Understand fix? (Yes: Edit | No: Search context, max 2 steps)
Edit → Correct? (Yes: Finish | No: Retry, max 1)
Repeated error? New approach.

=== COMMON MISTAKES ===
Format: Text after END; missing BEGIN; extra/dupe markers.
Code: \t/\n; mixed whitespace; string lines; no pre-read; huge content; wrong indent; off-by-one.
Workflow: Repeat reads/fails; no finish; endless explore; unnecessary verify.

=== REMINDERS ===
- Format precision critical.
- Finish promptly.
- Real whitespace only.

Available tools will be listed below."""

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history tree with unique ids
    - Builds the LLM context from the root to current node
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
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

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task) and an instruction node.
        system_prompt = SYSTEM_PROMPT
        self.system_message_id = self.add_message("system", system_prompt)
        self.user_message_id = self.add_message("user", "")
        self.instructions_message_id = self.add_message("instructor", "")
        
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])

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
                    print(self.message_id_to_context(msg_id))
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
                        continue
                    
                    # Get the function
                    func = self.function_map[function_name]
                    
                    # Execute the tool
                    try:
                        result = func(**arguments)
                        
                        # If finish was called, return the result
                        if function_name == "finish":
                            return result
                        
                        # Add tool result to tree
                        self.add_message("tool", str(result))
                        
                    except TypeError as e:
                        # Handle argument mismatch errors
                        import traceback
                        traceback.print_exc()
                        error_msg = f"Error calling {function_name}: {str(e)}"
                        self.add_message("tool", error_msg)
                    except Exception as e:
                        # Handle other execution errors
                        import traceback
                        traceback.print_exc()
                        error_msg = f"Error executing {function_name}: {str(e)}"
                        self.add_message("tool", error_msg)
                
                except ValueError as e:
                    # Handle parsing errors
                    import traceback
                    traceback.print_exc()
                    error_msg = f"Error parsing function call: {str(e)}"
                    self.add_message("tool", error_msg)
            
            except Exception as e:
                # Handle unexpected errors in the loop
                import traceback
                traceback.print_exc()
                error_msg = f"Unexpected error in step {step}: {str(e)}"
                self.add_message("tool", error_msg)
        
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