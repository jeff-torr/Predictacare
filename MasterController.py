from UserInterface import UserInterface
import sys


def main():
    isString = True
    while isString:
        name = input("Patients name? ")
        if type(name) == str:
            isString = False
        else:
            print("Please enter your name.")
            name = input("Patients name? ")

    isAge = True
    while isAge:
        age = input("Patients age? ")
        if type(age) == int:
            isAge = False
        else:
            print("Please enter your age.")
            age = input("Patients age? ")

    isGender = True
    while isGender:
        gender = input("Patients gender? ")
        if type(gender) == str:
            isGender = False
        else:
            print("Please enter your gender.")
            gender = input("Patients gender? ")
    
    isDescription = True
    while isDescription:
        description = input("Description of appointment: ")
        if type(description) == str:
            isDescription = False
        else:
            print("Please enter your description.")
            description = input("Description of appointment: ")

    isYN = True
    while isYN:
        familiarity = input("Are they are new patient? Y/N ")
        if familiarity == "y" or familiarity == "n" or familiarity == "Y" or familiarity == "N":
            isDescription = False
        else:
            print("Please enter your description.")
            familiarity = input("Are they are new patient? Y/N ")

    user = UserInterface(name, age, gender, description, familiarity)

if __name__ == "__main__":
  # only call main if we run this file directly
  # don't call main when we import this file
  main()