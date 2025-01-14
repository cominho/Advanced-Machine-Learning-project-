def _print(message, verbosity_level, current_verbose_level):
    """
    Helper function to handle conditional printing based on verbosity levels.

    Args:
        message (str): The message to print.
        verbosity_level (int): The level required for this message to be printed.
        current_verbose_level (int): The current verbosity level set by the user.
    """
    if current_verbose_level >= verbosity_level:
        print(message) 
def _print_error(message):
    """
    Helper function to print error messages. Always prints regardless of verbosity level.

    Args:
        message (str): The error message to print.
    """
    print(f"Error: {message}")