#15:28 - 15:35 - 15:40
import datetime
import math

def calculate_wave_frequency(age_in_days, cycle_length):
    return math.sin(2 * math.pi * age_in_days / cycle_length)

def get_age_in_days(birthdate):
    today = datetime.datetime.now()
    age = today - birthdate
    return age.days

def main():
    # Ask for user's information
    name = input("Enter your name: ")
    year = int(input("Enter the year of your birth (e.g., 1990): "))
    month = int(input("Enter the month of your birth (1-12): "))
    day = int(input("Enter the day of your birth (1-31): "))

    # Calculate birthdate
    birthdate = datetime.datetime(year, month, day)

    # Calculate age in days
    age_in_days = get_age_in_days(birthdate)

    # Calculate wave frequencies
    physical_wave = calculate_wave_frequency(age_in_days, 23)
    emotional_wave = calculate_wave_frequency(age_in_days, 28)
    intellectual_wave = calculate_wave_frequency(age_in_days, 33)

    # Print greeting and information
    print(f"Hello {name}!")
    print("Today's date is:", datetime.datetime.now().strftime("%Y-%m-%d"))
    print("Physical wave:", physical_wave)
    print("Emotional wave:", emotional_wave)
    print("Intellectual wave:", intellectual_wave)

    # Check if any wave is higher than 0.5 or lower than -0.5
    for wave, name in zip([physical_wave, emotional_wave, intellectual_wave], ["Physical", "Emotional", "Intellectual"]):
        if wave > 0.5:
            print(f"{name} wave is high! Congratulations!")
        elif wave < -0.5:
            print(f"{name} wave is low! Cheer up!")

    # Check if tomorrow's waves are better
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    tomorrow_age_in_days = get_age_in_days(birthdate) + 1
    tomorrow_physical_wave = calculate_wave_frequency(tomorrow_age_in_days, 23)
    tomorrow_emotional_wave = calculate_wave_frequency(tomorrow_age_in_days, 28)
    tomorrow_intellectual_wave = calculate_wave_frequency(tomorrow_age_in_days, 33)

    if tomorrow_physical_wave > physical_wave and tomorrow_emotional_wave > emotional_wave and tomorrow_intellectual_wave > intellectual_wave:
        print("Tomorrow will be better!")

if __name__ == "__main__":
    main()