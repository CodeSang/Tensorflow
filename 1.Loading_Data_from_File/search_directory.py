import os
import csv


def search(dirname):
    x_label = []
    y_label = []

    filenames = os.listdir(dirname)
    for filename in filenames:
        y_label.append(filename[:-6])
        full_filename = os.path.join(dirname, filename)
        x_label.append(full_filename)

    return x_label,y_label


if __name__ == "__main__" :

    x, y = search("../0.Image_Data/1.TrainImage")

    f = open('output.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)

    for i in range(len(x)):
        print(y[i])
        wr.writerow([x[i], y[i]])

    f.close()


