# import datetime
# import logging
#
# from colorama import Back, Fore, Style, init
#
# # Initialize Colorama for cross-platform color support
# init(autoreset=True)  # Resets color after each call
#
#
# class ColorLogFormatter(logging.Formatter):
#     """A custom log formatter that adds color and visual elements for better readability."""
#
#     COLOR_CODES = {
#         "DEBUG": Fore.CYAN + Style.DIM,
#         "INFO": Fore.GREEN,
#         "WARNING": Fore.YELLOW + Style.BRIGHT,
#         "ERROR": Fore.RED + Style.BRIGHT,
#         "CRITICAL": Fore.WHITE + Back.RED + Style.BRIGHT,  # Highlighting Critical
#         "DEFAULT": Style.RESET_ALL,  # Reset color
#     }
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def format(self, record):
#         log_color = self.COLOR_CODES.get(record.levelname, self.COLOR_CODES["DEFAULT"])
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         level_name = record.levelname.ljust(8)  # Pad level name for consistent width
#         message = super().format(record)
#
#         # Construct visually appealing log message
#         formatted_message = (
#             f"{log_color}{Style.BRIGHT}["
#             f"{Fore.WHITE}{timestamp}{log_color}"
#             f"] {level_name} - {message}{Style.RESET_ALL}"
#         )
#         return formatted_message
#
#
# class ColorLogger:
#     """A simple logger class that supports color and visually pleasing formatting."""
#
#     def __init__(self, name: str, log_file: str = None):
#         """
#         Initializes a new ColorLogger instance.
#
#         Args:
#             name (str): The name of the logger.
#             log_file (str, optional): The path to the log file. If None, logs are only printed to the console.
#         """
#         self.logger = logging.getLogger(name)
#         self.logger.setLevel(logging.DEBUG)  # Capture all levels
#
#         # Create a formatter
#         formatter = ColorLogFormatter("%(message)s")  # Message-only in formatter
#
#         # Create a console handler and set the formatter
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
#         self.logger.addHandler(console_handler)
#
#         # Create a file handler (optional)
#         if log_file:
#             file_handler = logging.FileHandler(log_file)
#             file_handler.setFormatter(
#                 logging.Formatter(
#                     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#                 )
#             )  # Simpler formatter for files
#             self.logger.addHandler(file_handler)
#
#     def debug(self, message: str):
#         """Logs a debug message."""
#         self.logger.debug(message)
#
#     def info(self, message: str):
#         """Logs an info message."""
#         self.logger.info(message)
#
#     def warning(self, message: str):
#         """Logs a warning message."""
#         self.logger.warning(message)
#
#     def error(self, message: str):
#         """Logs an error message."""
#         self.logger.error(message)
#
#     def critical(self, message: str):
#         """Logs a critical message."""
#         self.logger.critical(message)

import logging
import sys


class CustomLogger:
    def __init__(self, name, log_file):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )  # Simpler formatter for files
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)
