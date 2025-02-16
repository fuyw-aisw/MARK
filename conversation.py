#conversation
import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = []
            ret.append({'role':'system','content':self.system})
            for role, message in self.messages:

                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret.append({'role':self.roles,'content':message})
            #print(ret)
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])
    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
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
        
conv_v1 = Conversation(
    system="You are an AI assistant specializing in text induction. Please generate a topic name based on the provided input texts.",
    roles=("user"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep=" ",
    sep2="</s>",
)        

conv_v2 = Conversation(
    system="You are an AI assistant specializing in text generation. Please create a novel text based on the surrounding input texts.",
    roles=("user"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep=" ",
    sep2="</s>",
)

conv_v3 = Conversation(
    system="You are an AI assistant specializing in text prediction. Please identify the most likely cluster to which the given text belongs.",
    roles=("user"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep=" ",
    sep2="</s>",
)
#if __name__ == "__main__":
#print(conv_v1.get_prompt())
    
    
