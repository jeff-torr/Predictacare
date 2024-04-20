from UserInterface import UserInterface
import sys


<<<<<<< HEAD
def main():
    isString = True
    while isString:
        name = input("Patients name?")
        if type(name) == str:
            isString = False
        else:
            print("Please enter your name.")
=======
if __name__ == '__main__':
>>>>>>> refs/remotes/origin/main

    isAge = True
    while isAge:
        age = input("Patients age?")
        if type(age) == int:
            isAge = False
        else:
            print("Please enter your age.")

    isGender = True
    while isGender:
        gender = input("Patients gender?")
        if type(gender) == str:
            isGender = False
        else:
            print("Please enter your gender.")
    
    isDescription = True:
    while isDescription:
        description = input("Description of appointment:")
        if type(description) == str:
            isDescription = False
        else:
            print("Please enter your description.")

<<<<<<< HEAD
    isYN = True
    while isYN:
        familiarity = input("Are they are new patient? Y/N")
        if familiarity == "y" or familiarity == "n" or familiarity == "Y" or familiarity == "N":
            isDescription = False
        else:
            print("Please enter your description.")

    user = UserInterface(name, age, gender, description, familiarity)
=======
    user = UserInterface(name,age,gender,description, familiarity)

>>>>>>> refs/remotes/origin/main




