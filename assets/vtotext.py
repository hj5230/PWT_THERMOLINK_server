import re

def adjust_temperature(temperature):
    print(f"Command: Temperature adjusted to {temperature} degrees.")

def schedule_temperature_setting(temperature, hour, minute, period, day):
    print(f"Command: Temperature will be set to {temperature} degrees at {hour}:{minute} {period} on {day}.")

def change_temperature(from_temp, to_temp):
    print(f"Command: Changing temperature from {from_temp} degrees to {to_temp} degrees.")

def parse_commands(text):
    patterns = {
        r"(adjust the temperature to|set temperature to|make it|将温度调整到|设定温度为|温度改为) (\d+) degrees?度?": adjust_temperature,
        r"(set the temperature to|schedule temperature to|program the thermostat to|在([\w\s]+)的|预约|计划在([\w\s]+)的) (\d+):(\d+) (AM|PM) on ([\w\s]+)将温度设定为(\d+)度": schedule_temperature_setting,
        r"(change the temperature from|adjust from|move temperature from|从|温度从|将温度从) (\d+) degrees?度? to (\d+) degrees?度?": change_temperature,
    }

    for pattern, action in patterns.items():
        for pat in pattern.split('|'):
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                action(*match.groups())

# Example of using the function
text_to_parse = """
将温度调整到22度。
在明天上午9点将温度设定为25度。
从20度改变到25度。
设定温度为23度。
预约8:00 AM 星期一温度为24度。
温度从19度调到22度。
"""
text_to_parse = """
Adjust the temperature to 22 degrees.
Set temperature to 25 degrees at 9:00 AM on tomorrow.
Change the temperature from 20 degrees to 25 degrees.
Make it 23 degrees.
Program the thermostat to 24 degrees at 8:00 AM on Monday.
Move temperature from 19 degrees to 22 degrees.
"""
parse_commands(text_to_parse)
