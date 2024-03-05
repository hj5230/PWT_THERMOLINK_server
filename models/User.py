class User:
    def __init__(self, username, email, productId) -> None:
        self.username = username
        self.email = email
        self.productId = productId
    def to_dict(self) -> dict:
        return {
            "name": self.username,
            "email": self.email,
            "productId": self.productId
        }
