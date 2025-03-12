import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Union



class SeparatorStyle(Enum):
    """Different separator style."""
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    QWEN_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.MPT
    sep: str = "###"
    sep2: str = None
    version: str = "plain"

    skip_next: bool = False

    def get_prompt(self) -> str:
        """Generate the conversation prompt based on the separator style."""
        messages = self._prepare_messages()
        if self.sep_style == SeparatorStyle.MPT:
            return self._format_mpt(messages)
        elif self.sep_style == SeparatorStyle.PLAIN:
            return self._format_plain(messages)
        elif self.sep_style == SeparatorStyle.TWO:
            return self._format_two(messages)
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def _prepare_messages(self) -> List[Union[List[str], Tuple[str, Tuple[str, ...]]]]:
        """Prepare messages for formatting."""
        messages = self.messages
        if len(messages) > 0 and isinstance(messages[0][1], tuple):
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)
        return messages

    def _format_mpt(self, messages: List[Union[List[str], Tuple[str, Tuple[str, ...]]]]) -> str:
        """Format messages with MPT separator style."""
        ret = self.system + self.sep
        for role, message in messages:
            if message:
                if isinstance(message, tuple):
                    message, _, _ = message
                ret += role + message + self.sep
            else:
                ret += role
        return ret

    def _format_plain(self, messages: List[Union[List[str], Tuple[str, Tuple[str, ...]]]]) -> str:
        """Format messages with PLAIN separator style."""
        seps = [self.sep, self.sep2]
        ret = self.system
        for i, (role, message) in enumerate(messages):
            if message:
                if isinstance(message, tuple):
                    message, _, _ = message
                ret += message + seps[i % 2]
            else:
                ret += ""
        return ret

    def _format_two(self, messages: List[Union[List[str], Tuple[str, Tuple[str, ...]]]]) -> str:
        """Format messages with TWO separator style."""
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                if isinstance(message, tuple):
                    message, _, _ = message
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def append_message(self, role: str, message: str) -> None:
        """Append a message to the conversation."""
        self.messages.append([role, message])

    def copy(self) -> 'Conversation':
        """Create a copy of the conversation."""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version
        )

    def dict(self) -> dict:
        """Convert the conversation to a dictionary."""
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if isinstance(y, tuple) else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

    def get_images(self) -> List[str]:
        """Extract images from the messages."""
        images = []
        for _, message in self.messages:
            if isinstance(message, tuple):
                _, imgs, _ = message
                images.extend(imgs)
        return images



conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llama3 = Conversation(
    system="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

# conv_llama3 = Conversation(
#     system="You are a helpful AI assistant robot.",
#     roles=("USER", "ASSISTANT"),
#     version="llama3",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="<|end_of_text|>",
# )

conv_qwen2 = Conversation(
    system="You are a helpful AI assistant robot.",
    roles=("USER", "ASSISTANT"),
    version="qwen2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_templates = {
    "default": conv_llama3,
    "plain": conv_plain,
    "llama3": conv_llama3,
    "qwen2": conv_qwen2,
}

default_conversation = conv_llama3

if __name__ == "__main__":
    print(default_conversation.get_prompt())