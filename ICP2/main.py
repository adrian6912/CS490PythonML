if __name__=="__main__":
    def lbs_to_kg():
        """ICP part 1 converts a list of pounds to kilograms"""
        pounds = [150, 155, 145, 148]
        kilograms = [x * 0.45359237 for x in pounds]
        return kilograms

    def string_alternative():
        """ICP part 2 returns every other letter of a string"""
        string = "Good evening"
        return string[::2]

    def word_counter():
        """ICP part 3 counts the occurrences of individual words in a document"""
        counter = {}
        file = open('ICP2_Part3.txt', 'r+')
        lines = file.readlines()
        file.write('\n')
        for line in lines:
            words = line.split()
            for word in words:
                try:
                    counter[word] = counter[word] + 1
                except KeyError:
                    counter[word] = 1
        for word in counter.keys():
            file.write(word + ": " + str(counter[word]) + '\n')
        file.close()

    print(lbs_to_kg())
    print(string_alternative())
    word_counter()
