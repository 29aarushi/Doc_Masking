import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Extracts and formats the error message details including the file name, line number, and error message.

    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module to extract exception details.

    Returns:
        str: A formatted string containing the file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    """
    Custom exception class that formats and stores detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with a detailed error message.

        Args:
            error_message (str): The error message.
            error_detail (sys): The sys module to extract exception details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        """
        Returns the detailed error message as a string.

        Returns:
            str: The detailed error message.
        """
        return self.error_message
    


        