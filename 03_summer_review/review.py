wavelength = 193
NA = 1.35
name = "computational lithography"
is_coherent = True

print(wavelength)
print(NA)
print(name)
print(is_coherent)

print(type(wavelength))
print(type(NA))
print(type(name))
print(type(is_coherent))

a = 10
b = 3

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)

source_points = [-0.5, 0.0, 0.5]
print(source_points[0])
print(source_points[1])
print(source_points[2])

source_points = [-0.5, 0.0, 0.5]
for s in source_points:
    print(s)

fc = 1.0
frequency = 0.8
if abs(frequency) <= fc:
    print("Pass")
else:
    print("Blocked")

def is_inside_pupil(frequency, cutoff):
    return abs(frequency) <= cutoff
result = is_inside_pupil(0.8, 1.0)
print(result)
