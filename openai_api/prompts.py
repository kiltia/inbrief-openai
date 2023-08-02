CLASSIFY_TASK = "Представь, что ты классификатор. Я буду давай тебе текст, а ты отвечай мне названием класса к которой относится этот текст. В ответе выводи только его, без каких-либо дополнительных пояснений, извинений или иных слов."

CLASSIFY_EXAMPLE_REQUEST = """Во Франции продолжаются беспорядки. Вот главное:
▪️ В Нантере прошел траурный марш в память о 17-летнем парне, который был застрелен полицейским при отказе подчиниться их требованиям;
▪️ Постепенно марш перерос в массовые беспорядки — протестующие били витрины, жгли автомобили и здания, закидывали полицейских бутылками, петардами и камнями;
▪️ Также протестующие попытались прорваться к городской префектуре, полиция применила слезоточивый газ, для подавления беспорядков в город прибыл спецназ;
▪️ Беспорядки охватили многие крупные города Франции. Люди вышли на улицы в Париже, Лилле, Амьене, Сент-Этьене, Дижоне, Клермон-Ферране, Страсбурге и Лионе, сообщает Le Parisien;
▪️ Этой ночью МВД Франции задействует для подавления беспорядков 40 тысяч полицейских. Вчера их было 9 тысяч;
▪️ Макрон назвал массовые беспорядки «абсолютно неоправданными» и призвал граждан к спокойствию.
На видео: горящий офис банка Crédit Mutuelle в Нантере."""

CLASSIFY_EXAMPLE_RESPONSE = "Политика"

SUMMARIZE_TASK = "Тебе необходимо суммаризовать текст в новый текст, сохранив только САМЫЕ основные моменты повествования. Постарайся выделить основные тезисы в нем и оставить только их, сохранив основной сюжет текста."

SUMMARIZE_EXAMPLE_REQUEST = 'Советник главы Росэнергоатома Карчаа утверждает, что Киев в ночь на 5 июля якобы "собирается атаковать ЗАЭС с применением высокоточного оружия и беспилотников-камикадзе". (4 июля, 21:38).В Офисе президента Украины оценили угрозу теракта на Запорожской АЭС как "50 на 50". «С одной стороны россияне понимают: мир смотрит внимательно. И они понимают последствия юридического характера, в том числе для «Росатома» — они будут однозначно, если что-то Россия захочет сделать», — отметили в офисе. При этом Россия заявляет, что никаких терактов совершать на ЗАЭС не собирается и обвиняет украинскую сторону в желанию организовать провокацию. (4 июля, 21:13).Украина из-за «последних событий вокруг Запорожской АЭС» усиливает личным составом и техникой позиции на границе с Белоруссией. Об этом заявил командующий украинскими объединенными силами Сергей Наев. (4 июля, 19:02).Гендиректор МАГАТЭ Рафаэль Гросси: На этот раз АЭС удалось избежать полной потери всего внешнего электроснабжения, что уже случалось семь раз во время конфликта, но последнее отключение линии электропередачи вновь демонстрирует нестабильную ситуацию с ядерной безопасностью на АЭС. (4 июля 16:20).'

SUMMARIZE_EXAMPLE_RESPONSE = """
Ситуация вокруг ЗАЭС остается достаточно напряженной, несмотря на стабилизировавшуюся обстановку после седьмой по счету потери электроснабжения.

Украинская сторона допускает возможность теракта со стороны Российской Федерации, также поступают сообщения об усилении личного состава на границе с Белоруссией. 

В свою очередь представители Российской Федерации отвергли обвинения о возможном теракте, обвиняют Киев в провокациях и допускают атаку ЗАЭС уже в ночь на 5 июля со стороны ВСУ. 
"""

TITLE_EXAMPLE_REQUEST = SUMMARIZE_EXAMPLE_REQUEST

TITLE_EXAMPLE_RESPONSE = "Ситуация вокруг ЗАЭС"

TITLE_TASK = "Тебе будет дан текст. Тебе будет необходимо выделить общие черты сюжета в нем и придумать новостной заголовок для него в формате telegram заголовка. Отвечай только самим заголовком"
