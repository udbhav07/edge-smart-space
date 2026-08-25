from openai import OpenAI
import json

class SmartAgentLLM:
    def __init__(self, model_name="qwen2.5:7b"):
        # Connect to your local Ollama server
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model = model_name
        
    # System prompt: Tells the AI who it is and how to behave
        self.messages = [
            {"role": "system", "content": "You are Jarvis, a smart space assistant. Keep your spoken responses very brief, casual, and helpful."}
        ]
        
        # Define the tools (JSON Schema) that Qwen is allowed to use
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "control_appliance",
                    "description": "Turns a smart space appliance/device on or off.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device": {"type": "string", "description": "The name of the device (e.g., 'main lights', 'fan', 'ac')"},
                            "state": {"type": "string", "enum": ["on", "off"]}
                        },
                        "required": ["device", "state"]
                    }
                }
            }
        ]

    def _execute_tool(self, tool_name, arguments):
        """This function executes when the AI decides it needs to use a tool."""
        if tool_name == "control_appliance":
            device = arguments.get("device")
            state = arguments.get("state")
            
            # This is where your actual hardware/GPIO code will go later!
            print(f"\n[HARDWARE COMMAND]: Turning {state} the {device}...\n")
            
            return f"Success: The {device} is now {state}."
            
        return "Error: Unknown tool."

    def chat(self, user_text):
        """Sends user text to the LLM and handles any tool calls it wants to make."""
        self.messages.append({"role": "user", "content": user_text})
        
        # 1. Send the prompt and tools to Qwen
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools
        )
        
        msg = response.choices[0].message
        self.messages.append(msg)
        
        # 2. Did Qwen decide to use a tool?
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # Execute the python hardware function
                result = self._execute_tool(func_name, args)
                
                # Report the hardware result back to Qwen
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                
            # 3. Ask Qwen to generate a final spoken sentence (e.g., "I've turned off the lights.")
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            
            final_text = final_response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": final_text})
            return final_text
            
        else:
            # Qwen just answered normally without using tools
            self.messages.append({"role": "assistant", "content": msg.content})
            return msg.content