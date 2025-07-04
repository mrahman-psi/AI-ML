import hashlib
import os
import uuid
import datetime
import csv

class TaskManager:
    def __init__(self, userID, filename="tasks.csv"):
       self.userID = userID
       self.tasks = []
       print(f"File Name: {filename}. userName: {userID}")  


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

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()
def save_user_to_file(username, hashed_password, filename="users.txt"):
    """Saves the username and hashed password to a file."""
    with open(filename, mode="a") as file:
        file.write(f"{username},{hashed_password}\n")

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
                self.userID = username
                access_task_manager()
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
        access_task_manager()
        break

def save_task_to_csv(self, filename="tasks.csv"):
    # Define the header for the CSV file
    header = ["id", "date created", "description", "status", "userID"]
    
    # Open the CSV file in write mode
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        
        # Write the header
        writer.writeheader()
        
        # Write all expense data
        for task in self.tasks:
            writer.writerow(task)
    
    print(f"Task has been saved to {filename}")

# Lead task data from existing file
def load_task_from_csv(self, filename="tasks.csv"):
    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            
            # Load each row as a dictionary and append to the task list
            self.tasks = []
            for row in reader:         
                self.tasks.append(row)
        print(f"Task have been loaded from {filename}")
    except FileNotFoundError:
        print(f"No previous data found. Starting fresh.")

def add_task(self, description=""):
    # Generate a unique ID
    unique_id = uuid.uuid4()
    userID = userID
    aTask = {

        "id": unique_id,
        "date created": datetime.now().strftime("%Y-%m-%d"),
        "description": description,
        "status": "Pending",
        "UserID": userID          
    }
    self.tasks.append(aTask)
    # Save the task to a file
    print(f"Task added: {aTask}")

def view_tasks(self):
    if not self.tasks:
        print("No task found.")
        return
    for task in self.tasks:
        print(f"{task['id']} | {task['date created']} | ${task['description']} | {task['status']} | {task['UserID']}")


def access_task_manager():
    # Initialize task manager
    #tm = TaskManager()   

    while True:
        print("\nOptions: ")
        print("1. Add a Task")
        print("2. View Tasks")
        print("3. Mark a Task as Completed")
        print("4. Delete a Task")    
        print("5. Logout")
        
        choice = input("Choose an option (1-5): ")

        if choice == '1':
            task_description = input("Enter description (optional): ")
            add_task(task_description)
        elif choice == '2':
            view_tasks()
        elif choice == '3':
            #tm.track_budget()
            print("Mark a Task as Completed")
        elif choice == '4':
            #tm.save_expenses_to_csv()
            print("Delete a Task")  
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please choose again.")


# Example usage
if __name__ == "__main__":
    login_to_task_manager()
