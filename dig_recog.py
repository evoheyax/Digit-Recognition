from __future__ import print_function, division
import pattern_recog_func as prf
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == "__main__":
    dig_data = load_digits()
    X = dig_data.data
    y = dig_data.target

    # PART A
    print("---------------------")
    print("Part A...")
    print("---------------------")
    
    md_clf = prf.svm_train(X[0:60], y[0:60])

    md_predictions = md_clf.predict(X[61:81])
    md_targets = y[61:81]
    
    print("Predictions: {}".format(md_predictions))
    print("Targets:     {}".format(md_targets))

    total = len(md_targets)
    incorrect = 0
    for i in range(total):
        if(md_predictions[i] != md_targets[i]):
            incorrect += 1
            print("--------> index, actual digit, svm_prediction: {} {} {}".format(60+i, md_targets[i], md_predictions[i]))

    perc_correct = (total-incorrect)/total

    print("Total number of mid-identifications: {}".format(incorrect))
    print("Success rate: {}".format(perc_correct))

    # PART B
    print("---------------------")
    print("Part B...")
    print("---------------------")

    dig_img = dig_data.images

    unseen = mpimg.imread("unseen_dig.png")[:, :, 0]
    unseen_interpoled, unseen_interpoled_flat = prf.interpol_im(unseen, plot_new_im = True)

    plt.grid(True)
    plt.imshow(dig_img[15], cmap="Greys")
    plt.show()

    unseen_interpoled_rescaled = prf.rescale_pixel(unseen_interpoled)

    print("Prediction (without rescaling): {}".format(md_clf.predict(unseen_interpoled.reshape(1, -1))[0]))
    print("Prediction (with rescaling):    {}".format(md_clf.predict(unseen_interpoled_rescaled.reshape(1, -1))[0]))
    print("Correct Number:                 5")
