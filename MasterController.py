from UserInterface import UserInterface
import sys


def main():

    name = input("Patients name?")
    age = input("Patients age?")
    gender = input("Patients gender?")
    description = input("Description of appointment:")
    familiarity = input("Are they are new patient? Y/N")


    user = UserInterface(name,age,gender,description)





