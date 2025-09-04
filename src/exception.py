## Creating Custom exception over here

import sys

def error_message_detail(error,error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script [{file_name}] "
            f"at line [{line_number}] "
            f"with message: {str(error)}"
        )
    else:
        error_message = f"Error: {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    


