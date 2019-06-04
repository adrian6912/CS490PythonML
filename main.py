if __name__ == "__main__":

    def replacer(string):
        """Program 2a from ICP1"""
        listy = string.split()
        newList = []
        for elmt in listy:
            if elmt.lower() == 'python':
                newList.append('pythons')
            else:
                newList.append(elmt)
        newString = str()
        for elmt in newList:
            newString = newString + " " + elmt
        return newString.strip()

    def delCharacter(string, char):
        """Deletes a specified character from a given string"""
        ind = string.find(char)
        if ind == -1: #Character isn't in the original string, return original string
            return string
        newString = string[:ind] + string[ind + 1:]
        return newString

    def play():
        """Program 3 from ICP 1"""
        string = input("Please input the word Python")
        newString = delCharacter(string, 'n')
        newString = delCharacter(newString, 'h')
        reversedString = newString[::-1]
        print(reversedString)

    def arith():
        """Program 2b from ICP1"""
        x = int(input())
        y = int(input())
        print(x+y)

    play()
    arith()
    x = input()
    print(replacer(x))

