import pandas

def main():
    train_path = './dataset/train.json'
    test_path = './dataset/test.json'
    train_data = pandas.read_json(train_path)
    test_data = pandas.read_json(test_path)
    print(train_data)

if __name__ == '__main__':
    main()