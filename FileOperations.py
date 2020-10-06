
def store_m(file, values):
    result = str(values[0])
    for i in range(1, len(values)):
        result += " "+str(values[i])
    result += "\n"
    file.write(result)
