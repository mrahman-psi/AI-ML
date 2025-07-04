import hashlib
import os

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()
def save_user_to_file(username, hashed_password, filename="users.txt"):
    """Saves the username and hashed password to a file."""
    with open(filename, mode="a") as file:
        file.write(f"{username},{hashed_password}\n")

def validate_credentials(username, password, filename="users.txt"):
    """Validates the username and password against stored data."""
    if not os.path.exists(filename):
        print("Can't find the file.")
        return False
    
    hashed_password = hash_password(password)
    
    with open(filename, mode="r") as file:
        for line in file:
            stored_username, stored_hashed_password = line.strip().split(",")
            if username == stored_username and hashed_password == stored_hashed_password:
                return True
    return False
def is_username_unique(username, filename="users.txt"):
    """Checks if the username is unique in the file."""
    if not os.path.exists(filename):
        return True  # If the file doesn't exist, all usernames are unique
    
    with open(filename, mode="r") as file:
        for line in file:
            existing_username, _ = line.strip().split(",")
            if username == existing_username:
                return False
    return True

def login_to_task_manager(filename="users.txt"):
    """Prompts the user to log in and grants access upon successful validation."""
    print("Welcome to the Task Manager!")
    while True:
        username = input("Enter your username: ").strip()
        #Validate if the username exists
        if not is_username_unique(username, filename):
            print("Welcome Back, please enter your password.")
            password = input("Enter your password: ").strip()        
            if validate_credentials(username, password, filename):
                print(f"Login successful! Welcome, {username}.")
                break
            else:
                print("Invalid username or password. Please try again.")
            continue
        print(f"Welcome to Task Manager {username}. Let's setup a password.")
        password = input("Enter a password: ").strip()
        confirm_password = input("Confirm your password: ").strip()
        
        if password != confirm_password:
            print("Passwords do not match. Please try again.")
            continue
        
        hashed_password = hash_password(password)
        save_user_to_file(username, hashed_password, filename)
        print("User registered successfully!")
        break


# Example usage
if __name__ == "__main__":
    login_to_task_manager()
