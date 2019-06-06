if __name__=="__main__":
    def lbs_to_kg():
        """ICP part 1 converts a list of pounds to kilograms"""
        weights_in = input("Give me the weight in pounds!\n")
        pounds = weights_in.split(',')
        kilograms = [int(x) * 0.45359237 for x in pounds]
        return kilograms

    def string_alternative():
        """ICP part 2 returns every other letter of a string"""
        string = input("Input a sentence\n")
        return string[::2]

    def word_counter():
        import string as s
        """ICP part 3 counts the occurrences of individual words in a document"""
        counter = {}
        file = open('ICP2_Part3.txt', 'r+')
        lines = file.readlines()
        file.write('\n')
        for line in lines:
            words = line.split()
            for word in words:
                stripped_word = word.strip(s.punctuation)
                try:
                    counter[stripped_word.lower()] = counter[stripped_word.lower()] + 1
                except KeyError:
                    counter[stripped_word.lower()] = 1
        for word in counter.keys():
            file.write(word + ": " + str(counter[word]) + '\n')
        file.close()

    print(lbs_to_kg())
    print(string_alternative())
    word_counter()
