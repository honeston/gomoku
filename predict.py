import keras
import numpy as np

model = keras.models.load_model('model.h5', compile=False)

# predict_classes = model.predict(x_test[1:11,], batch_size=32)
# true_classes = np.argmax(y_test[1:11],1)
# print(confusion_matrix(true_classes, predict_classes))

predict = model.predict(np.asarray([[1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,
1,1,1,1,0,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,
1,1,1,1,2,1,1,1,1,1,
1,1,1,1,2,1,1,1,1,1,
1,1,1,1,2,1,1,1,1,1,
1,1,1,1,2,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1]]))

print(predict)
