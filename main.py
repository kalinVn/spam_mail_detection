from factory.Service import Service


def spam_mail_prediction():
    factory_service = Service()
    service = factory_service.get_service()

    service.preprocess()
    service.build()
    service.test_accuracy_score()

    input_mail = ["Nah I don't think he goes to usf, he lives around here though"]

    prediction = service.predict(input_mail)



spam_mail_prediction()






