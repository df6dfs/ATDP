from transformers import RobertaForMaskedLM, BartForConditionalGeneration, T5ForConditionalGeneration

class RobertaForPrompt(RobertaForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        parser.add_argument("--init_answer_words", type=int, default=1, )
        parser.add_argument("--init_type_words", type=int, default=1, )
        parser.add_argument("--init_answer_words_by_one_token", type=int, default=0, )
        parser.add_argument("--use_template_words", type=int, default=1, )
        parser.add_argument("--arc_s", type=float, default=64.0, help="ArcFace scale factor")
        parser.add_argument("--arc_m", type=float, default=0.5, help="ArcFace margin")
        return parser

class BartRE(BartForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser
        
class T5RE(T5ForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser