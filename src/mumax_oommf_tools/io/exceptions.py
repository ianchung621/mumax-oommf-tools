class OVF2Error(Exception):

    def __init__(self, filename: str, message: str):
        self.filename = filename
        self.message = message
        super().__init__(f"Error parsing {filename}: {message}")