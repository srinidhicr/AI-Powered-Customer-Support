#!/usr/bin/env python3
"""
Terminal-based interactive chatbot for Customer Support Copilot.
Features:
  - Persistent conversation memory
  - Context-aware follow-ups
  - Tool invocation visibility
  - Token usage tracking
  - Category-aware ticket management

Run: python cli_chatbot.py
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Suppress verbose LangChain output
os.environ['LANGCHAIN_VERBOSE'] = '0'

@dataclass
class Message:
    """Represents a single message in conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str
    tool_name: Optional[str] = None  # e.g., 'classify', 'retrieve', 'generate'
    metadata: Optional[Dict] = None  # e.g., {'confidence': 0.85, 'category': 'Technical'}
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ConversationSession:
    """Manages a multi-turn conversation with context."""
    ticket_id: str
    category: str
    created_at: str
    messages: List[Message]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def add_message(self, role: str, content: str, tool_name: str = None, metadata: Dict = None):
        """Add a message to conversation history."""
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            metadata=metadata or {}
        )
        self.messages.append(msg)
    
    def get_context_str(self, max_messages: int = 5) -> str:
        """Build context string from recent messages for LLM."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        context_lines = []
        for msg in recent:
            if msg.role == 'user':
                context_lines.append(f"[CUSTOMER]: {msg.content}")
            elif msg.role == 'assistant' and msg.tool_name is None:
                context_lines.append(f"[AGENT]: {msg.content[:200]}")
        
        return "\n".join(context_lines)
    
    def to_dict(self):
        return {
            'ticket_id': self.ticket_id,
            'category': self.category,
            'created_at': self.created_at,
            'messages': [m.to_dict() for m in self.messages],
            'metadata': self.metadata
        }


class CliChatbot:
    """Terminal-based chatbot interface with memory."""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize CLI chatbot.
        
        Args:
            use_cache: Enable semantic cache for repeated queries
        """
        self.use_cache = use_cache
        self.session: Optional[ConversationSession] = None
        self.session_file = 'data/sessions/current_session.json'
        
        # Import here to avoid LLM loading during module init
        from src.agents.orchestrator import run, get_final_draft
        from src.models.classifier import SupportClassifier
        
        self.run_orchestrator = run
        self.get_final_draft = get_final_draft
        self.classifier = SupportClassifier()
        
        os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
    
    def _print_header(self):
        """Print welcome header."""
        print("\n" + "="*70)
        print("  🤖 Customer Support Copilot — Terminal Interface")
        print("="*70)
        print("  Commands:")
        print("    /new        → Start new ticket")
        print("    /info       → Show ticket info")
        print("    /history    → Show conversation history")
        print("    /clear      → Clear current session")
        print("    /save       → Save session to file")
        print("    /load       → Load previous session")
        print("    /quit       → Exit")
        print("="*70 + "\n")
    
    def _print_separator(self, char: str = "─", width: int = 70):
        """Print a separator line."""
        print(char * width)
    
    def _get_color(self, role: str) -> str:
        """Get ANSI color code for role."""
        colors = {
            'user': '\033[94m',        # Blue
            'assistant': '\033[92m',   # Green
            'system': '\033[93m',      # Yellow
            'error': '\033[91m',       # Red
        }
        reset = '\033[0m'
        return colors.get(role, '') + '{}' + reset
    
    def new_ticket(self):
        """Create a new support ticket."""
        print("\n" + self._print_separator() + "\n")
        print("📝 Starting new ticket...")
        print("Describe your issue (or /quit to cancel):\n")
        
        query = input(">>> ").strip()
        if query == '/quit':
            return
        
        if not query:
            print("❌ Empty query. Try again.")
            return
        
        # Classify the query to determine category
        print("\n⏳ Analyzing your query...")
        clf_result = self.classifier.predict(subject='', body=query, tags=None)
        
        category = clf_result['category']
        confidence = clf_result['confidence']
        
        # Create session
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        self.session = ConversationSession(
            ticket_id=ticket_id,
            category=category,
            created_at=datetime.now().isoformat(),
            messages=[]
        )
        
        # Add initial message
        self.session.add_message('user', query)
        
        # Show classification
        print(f"\n📌 Ticket created: {ticket_id}")
        print(f"   Category: {category} (confidence: {confidence:.1%})")
        
        # Generate response
        self._generate_response(query)
    
    def _generate_response(self, query: str):
        """Generate and display AI response."""
        if not self.session:
            print("❌ No active ticket. Use /new to create one.")
            return
        
        print("\n⏳ Thinking...")
        start = time.time()
        
        # Build context from conversation history
        context = self.session.get_context_str(max_messages=3)
        enriched_query = query
        if context:
            enriched_query = f"{query}\n\n[Prior context: {context}]"
        
        try:
            result = self.run_orchestrator(enriched_query, use_cache=self.use_cache)
            elapsed = time.time() - start
            
            draft = self.get_final_draft(result)
            cache_hit = result.get('cache_hit', False)
            
            if not draft:
                draft = "❌ I was unable to generate a response. Please try rephrasing your query."
            
            # Display response
            print(f"\n{self._get_color('assistant').format('='*70)}")
            print(f"\n{self._get_color('assistant').format('SUGGESTED RESPONSE:')}")
            print(f"\n{draft}")
            print(f"\n{self._get_color('assistant').format('='*70)}")
            print(f"⏱️  Generated in {elapsed:.2f}s {' ⚡ (cached)' if cache_hit else ''}")
            
            # Store in session
            self.session.add_message('assistant', draft, metadata={
                'elapsed_sec': elapsed,
                'cache_hit': cache_hit
            })
            
            # Show tool visibility (for debugging)
            if '--debug' in sys.argv:
                self._show_tool_calls(result)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.session.add_message('system', f"Error: {str(e)}")
    
    def _show_tool_calls(self, result: dict):
        """Show tool calls made by orchestrator (debug mode)."""
        print("\n" + self._print_separator("─") + "\n")
        print("🔧 DEBUG: Tool calls made")
        
        messages = result.get('messages', [])
        tool_count = sum(1 for msg in messages if hasattr(msg, 'tool_calls'))
        print(f"   Total messages: {len(messages)}")
        print(f"   Tool invocations: {tool_count}")
        
        for i, msg in enumerate(messages[-5:]):  # Show last 5
            content = getattr(msg, 'content', '')
            content_type = type(content).__name__
            preview = str(content)[:80].replace('\n', ' ')
            print(f"   [{i}] {type(msg).__name__} | {preview}...")
    
    def show_ticket_info(self):
        """Display current ticket information."""
        if not self.session:
            print("❌ No active ticket.")
            return
        
        print("\n" + self._print_separator() + "\n")
        print(f"🎫 Ticket ID: {self.session.ticket_id}")
        print(f"📂 Category: {self.session.category}")
        print(f"📅 Created: {self.session.created_at[:10]}")
        print(f"📊 Messages: {len(self.session.messages)}")
        print(f"⏱️  Duration: {self._get_duration()}")
        
        # Show tags/metadata if any
        if self.session.metadata:
            print(f"\nℹ️  Metadata: {json.dumps(self.session.metadata, indent=2)}")
    
    def _get_duration(self) -> str:
        """Calculate session duration."""
        if not self.session or not self.session.messages:
            return "N/A"
        
        created = datetime.fromisoformat(self.session.created_at)
        now = datetime.now()
        delta = (now - created).total_seconds()
        
        if delta < 60:
            return f"{delta:.0f}s"
        elif delta < 3600:
            return f"{delta/60:.1f}m"
        else:
            return f"{delta/3600:.1f}h"
    
    def show_history(self):
        """Display conversation history."""
        if not self.session or not self.session.messages:
            print("❌ No messages in current session.")
            return
        
        print("\n" + self._print_separator() + "\n")
        print(f"📜 Conversation History ({len(self.session.messages)} messages)\n")
        
        for i, msg in enumerate(self.session.messages, 1):
            role = msg.role.upper()
            color = self._get_color(msg.role)
            
            # Truncate long content
            content = msg.content
            if len(content) > 120:
                content = content[:120] + "..."
            
            print(f"{i}. {color.format(f'[{role}]')} {content}")
            if msg.tool_name:
                print(f"   └─ Tool: {msg.tool_name}")
            print()
    
    def save_session(self):
        """Save current session to JSON file."""
        if not self.session:
            print("❌ No active ticket to save.")
            return
        
        data = self.session.to_dict()
        
        os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
        with open(self.session_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Session saved to {self.session_file}")
    
    def load_session(self):
        """Load previous session from file."""
        if not os.path.exists(self.session_file):
            print(f"❌ No saved session found at {self.session_file}")
            return
        
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct session
            self.session = ConversationSession(
                ticket_id=data['ticket_id'],
                category=data['category'],
                created_at=data['created_at'],
                messages=[],
                metadata=data.get('metadata', {})
            )
            
            # Reconstruct messages
            for msg_data in data['messages']:
                msg = Message(
                    role=msg_data['role'],
                    content=msg_data['content'],
                    timestamp=msg_data['timestamp'],
                    tool_name=msg_data.get('tool_name'),
                    metadata=msg_data.get('metadata')
                )
                self.session.messages.append(msg)
            
            print(f"✅ Loaded session: {self.session.ticket_id}")
            print(f"   Category: {self.session.category}")
            print(f"   Messages: {len(self.session.messages)}")
        
        except Exception as e:
            print(f"❌ Failed to load session: {e}")
    
    def clear_session(self):
        """Clear current session."""
        if not self.session:
            print("❌ No active session.")
            return
        
        confirm = input("⚠️  Clear session? (y/n): ").strip().lower()
        if confirm == 'y':
            self.session = None
            print("✅ Session cleared.")
        else:
            print("Cancelled.")
    
    def run(self):
        """Main REPL loop."""
        self._print_header()
        
        while True:
            # Prompt with ticket ID if active
            if self.session:
                prompt = f"\n[{self.session.ticket_id}] >>> "
            else:
                prompt = "\n>>> "
            
            try:
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    # Regular message
                    if not self.session:
                        print("❌ No active ticket. Use /new to create one.")
                        continue
                    
                    self.session.add_message('user', user_input)
                    self._generate_response(user_input)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
    
    def _handle_command(self, cmd: str):
        """Handle special commands."""
        cmd = cmd.lower().strip()
        
        if cmd == '/new':
            self.new_ticket()
        elif cmd == '/info':
            self.show_ticket_info()
        elif cmd == '/history':
            self.show_history()
        elif cmd == '/clear':
            self.clear_session()
        elif cmd == '/save':
            self.save_session()
        elif cmd == '/load':
            self.load_session()
        elif cmd == '/quit' or cmd == '/exit':
            print("\n👋 Goodbye!")
            sys.exit(0)
        elif cmd == '/help':
            self._print_header()
        else:
            print(f"❌ Unknown command: {cmd}. Use /help for options.")


def main():
    """Entry point."""
    chatbot = CliChatbot(use_cache=True)
    chatbot.run()


if __name__ == '__main__':
    main()