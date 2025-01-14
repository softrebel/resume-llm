from src.llm_handlers.gemini_handler import BaseLLMHandler, GeminiHandler
from google.ai.generativelanguage_v1beta.types import content
from src._core import project_configs
from src.resume.schema import ResumeBase, ResumeGenerated
import json
import markdown
import os
import pdfkit


def setup_resume_llm(resume: ResumeBase):
    default_generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            enum=[],
            required=["bio", "experiences", "skills"],
            properties={
                "bio": content.Schema(
                    type=content.Type.STRING,
                ),
                "experiences": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
                "skills": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
                "courses": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
                "educations": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
                "projects": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
                "publications": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
            },
        ),
        "response_mime_type": "application/json",
    }
    chat_history = [
        {
            "role": "user",
            "parts": [
                "you are excelent resume builder. i want to send my resume step by step based on section. consider my writing and listen carefully  and dont answer until i say.",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Understood. I'm ready. I will listen carefully to each section of your resume as you provide it and won't respond until you tell me you're finished. I'll be focusing on clarity, impact, and overall effectiveness as you go. Let's begin whenever you're ready.\n",
            ],
        },
        {
            "role": "user",
            "parts": [resume.courses],
        },
        {
            "role": "model",
            "parts": [
                "Okay, I have carefully noted all the training courses you've provided, including the course names, providers, locations, years, durations, and certificate availability. I'm ready for the next section when you are.\n",
            ],
        },
        {
            "role": "user",
            "parts": [resume.experiences],
        },
        {
            "role": "model",
            "parts": [
                "Okay, I've carefully recorded your work experience, including job titles, employers, technologies used, employment type, locations, and start/end dates. I've also noted the \"present\" indicator where applicable. I'm ready for the next section of your resume when you are.\n",
            ],
        },
        {
            "role": "user",
            "parts": [resume.skills],
        },
        {
            "role": "model",
            "parts": [
                "Okay, I have meticulously recorded your skills, including the subject of the skill, the specific tools, and your proficiency level for each. I am ready for the next section of your resume.\n",
            ],
        },
        {
            "role": "user",
            "parts": [resume.projects],
        },
        {
            "role": "model",
            "parts": [
                "Okay, I've meticulously recorded your projects, publications, and teaching experience, along with their associated tools, dates, and relevant links (where provided). This detailed list provides a strong overview of your practical experience and capabilities. I'm ready for the final sections of your resume when you are.\n",
            ],
        },
        {
            "role": "user",
            "parts": [resume.educations],
        },
        {
            "role": "model",
            "parts": [
                "Okay, I've meticulously recorded your projects, publications, and teaching experience, along with their associated tools, dates, and relevant links (where provided). This detailed list provides a strong overview of your practical experience and capabilities. I'm ready for the final sections of your resume when you are.\n",
            ],
        },
        {
            "role": "user",
            "parts": [resume.publications],
        },
        {
            "role": "model",
            "parts": [
                "Okay, I have recorded the titles and links to your published books. This completes all the sections of your resume that you've provided. I'm now ready to give you feedback on the overall structure and content, as well as suggestions for improvement. Please let me know when you are ready for me to proceed with that feedback.\n",
            ],
        },
        {
            "role": "user",
            "parts": [
                "Now I send you a job description. you customize and prepare my resume suitable for job description. consider that you must select information that highly related. you can add and edit text to relate correctly.",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Understood. Please provide the job description. I will carefully analyze it and then tailor your resume, selecting the most relevant information, adjusting wording, and adding details as necessary to align your qualifications with the specific requirements of the job. I'm ready when you are.\n",
            ],
        },
        {
            "role": "user",
            "parts": [
                "and the result must be in persian language\n",
            ],
        },
        {
            "role": "model",
            "parts": [
                "okay.",
            ],
        },
        # {
        #     "role": "user",
        #     "parts": [
        #         "·         استخراج روش‌های مختلف مدل‌سازی از طریق مطالعه منابع معتبر مرتبط با ریسک اعتباری·         استفاده از علوم کامپیوتر و آمار در کنار دانش موجود در حوزه کاری سازمان به منظور تحلیل داده‌ها·         استفاده از ابزارها و روش‌های مختلف به منظور آماده سازی و استخراج شاخص‌ها، دستیابی به اطلاعات و بینش مورد نیاز سازمان از طریق تحلیل حجم عظیمی از داده‌ها با استفاده از روش‌های مبتنی بر هوش مصنوعی و یادگیری ماشین ·         مسلط بر زبان برنامه نویسی پایتون·         ارائه اطلاعات استخراج‌شده در قالت گزارش‌های مدیریتی·         دارای حداقل دو سال سابقه کار مرتبط",
        #     ],
        # },
        # {
        #     "role": "model",
        #     "parts": [
        #         "بسیار خب، من شرح شغل را دریافت کردم و با دقت آن را بررسی کردم. اکنون، رزومه شما را با تمرکز بر موارد کلیدی ذکر شده در شرح شغل، به زبان فارسی، بهینه سازی می کنم. در این فرآیند، اطلاعات مرتبط را برجسته کرده و در صورت نیاز، متن را ویرایش و تکمیل می کنم تا رزومه شما به بهترین شکل با الزامات این موقعیت شغلی هماهنگ شود.\n\n**رزومه بهینه شده برای موقعیت شغلی تحلیلگر ریسک اعتباری (با تاکید بر هوش مصنوعی و یادگیری ماشین):**\n\n**1. اطلاعات شخصی:**\n\n*   نام و نام خانوادگی: (نام شما)\n*   شماره تماس: (شماره تماس شما)\n*   ایمیل: (ایمیل شما)\n*   لینک گیت هاب (اختیاری): (لینک گیت هاب شما)\n\n**2. خلاصه:**\n\nمتخصص با تجربه در زمینه توسعه نرم افزار و تحلیل داده، با تمرکز ویژه بر کاربرد هوش مصنوعی و یادگیری ماشین در تحلیل داده های حجیم. دارای سابقه موفق در استخراج بینش های ارزشمند از داده ها، مدل سازی و ارائه گزارش",
        #     ],
        # },
        # {
        #     "role": "user",
        #     "parts": [
        #         "·         استخراج روش‌های مختلف مدل‌سازی از طریق مطالعه منابع معتبر مرتبط با ریسک اعتباری·         استفاده از علوم کامپیوتر و آمار در کنار دانش موجود در حوزه کاری سازمان به منظور تحلیل داده‌ها·         استفاده از ابزارها و روش‌های مختلف به منظور آماده سازی و استخراج شاخص‌ها، دستیابی به اطلاعات و بینش مورد نیاز سازمان از طریق تحلیل حجم عظیمی از داده‌ها با استفاده از روش‌های مبتنی بر هوش مصنوعی و یادگیری ماشین ·         مسلط بر زبان برنامه نویسی پایتون·         ارائه اطلاعات استخراج‌شده در قالت گزارش‌های مدیریتی·         دارای حداقل دو سال سابقه کار مرتبط\n",
        #     ],
        # },
        # {
        #     "role": "model",
        #     "parts": [
        #         '```json\n{\n  "bio": "متخصص با تجربه در زمینه توسعه نرم افزار و تحلیل داده، با تمرکز ویژه بر کاربرد هوش مصنوعی و یادگیری ماشین در تحلیل داده های حجیم. دارای سابقه موفق در استخراج بینش های ارزشمند از داده ها، مدل سازی و ارائه گزارش های مدیریتی.",\n  "experiences": [\n    "مهندس نرم افزار در سازمان بهداشت و درمان صنعت نفت (1396-1397): توسعه سیستم های نرم افزاری با استفاده از WCF و ASP.NET MVC.",\n    "توسعه دهنده Python در سپهر پیام تدبیر ایرانیان (1396-1397): توسعه سیستم های جمع آوری داده با استفاده از Python، Scrapy و Crawl برای شبکه های اجتماعی.",\n    "توسعه دهنده وب Backend در سپهر پیام تدبیر ایرانیان (1397-1398): توسعه سیستم های وب با استفاده از PHP، CodeIgniter و MySQL.",\n    "توسعه دهنده وب Backend در موسسه رشد و سرمایه انسانی ایرانیان (رسا) (1398-1400): توسعه API های RESTful با استفاده از PHP، Laravel و MySQL.",\n    "توسعه دهنده وب Backend در سامانه خرید آنلاین بیمه (آلبیم) (1400-1401): توسعه سیستم های وب با استفاده از .net core5، MSSQL Server و React Js.",\n     "توسعه دهنده وب Backend در دانش گستر سینا (1402): توسعه سیستم های داده با استفاده از Python, FastAPI, Celery, requests, scrapy.",\n    "مدرس زبان‌های برنامه نویسی و هوش مصنوعی در مجتمع فنی نیاوران (1399-تاکنون): تدریس Python، Data Analysis، Machine Learning و Deep Learning.",\n    "طراح سامانه‌های هوشمند مدیریت سرمایه انسانی در پژوهشکده سرمایه انسانی دافوس آجا (1402-1403): توسعه سیستم های هوشمند با استفاده از Python، Tensorflow و Pytorch."\n  ],\n  "skills": [\n    "مسلط به زبان برنامه نویسی Python",\n    "مسلط به کتابخانه های پایتون برای تحلیل داده و یادگیری ماشین (Numpy, Pandas, Scikit-learn, Tensorflow, Pytorch)",\n    "تجربه در توسعه API های RESTful با استفاده از FastAPI و Laravel",\n    "تجربه در کار با پایگاه داده های SQL و NoSQL (MySQL, SQL Server, NEO4J)",\n    "تجربه در جمع آوری داده از وب با استفاده از Scrapy و Selenium",\n    "تجربه در پیاده سازی مدل های یادگیری ماشین و هوش مصنوعی",\n    "تجربه در ارائه گزارش های مدیریتی و تحلیل داده ها"\n  ],\n  "courses": [\n      "Network+",\n      "CCNA",\n      "MCSA 2012",\n      "SPSS",\n      "SQL Server",\n      "Web Application Vulnerabilities and Security Attacks",\n      "Modern PHP"\n  ],\n    "projects": [\n    "پیاده‌سازی وب سرویس از روی مدل برداری ساخته شده از کلمات ویکی‌پدیا فارسی با الگوریتم word2vec (Python, Word Embedding , Flask, Word2Vec)",\n    "سیستم مدیریت قرارداد پزشکان سازمان بهداشت و درمان صنعت نفت (WCF, EntityFramework Code First)",\n     "سیستم رصد و تحلیل اخبار فرهنگی: کراول و پردازش محتوای وبسایت های خبری (PHP, CodeIgniter, Python, NLP)",\n     "کراولر جابینجا بهمراه بات تلگرامی دریافت آگهی‌ها (Python, Telegram Bot API)",\n     "سیستم بات اینستاگرام، پیدا کردن id از روی شماره افراد، مدیریت اتوماتیک فالو و آنفالو (Python, API Instagram)",\n     "سیستم کراول maps google :جمع آوری نقاط مکانی بهمراه مشخصات (Python, Selenium, BeautifulSoup)",\n     "سیستم کانون ارزیابی کارکنان (Assessment Center) (PHP, Laravel web API, vue.js)",\n     "سیستم آزمون/ گزینش آنلاین رسا (PHP, Laravel web API, vue.js)",\n     "کراول فیلم‌های ایرانی دهه 90 فیلیمو و ذخیره و بازیابی در دیتابیس گرافی NEO4J (Python, NEO4J Graph Database)",\n     "سیستم مشاوره و خرید بیمه آنلاین (.net core 5 web API, react.js)",\n     "استخراج داده از سایت ایرانداک (Python, scrapy)",\n     "تحلیل شبکه ارتباطی مقالات ایرانداک (Python, igraph)",\n     "MapReduce sample project using mrjob (Python, mrjob)",\n     "PySpark Machine Learning Project (Python, pyspark)",\n     "A Comparison of Centrality Measures for Community Evolution Discovery (Python, igraph, GED)",\n     "وب‌سرویس لاگین و دریافت فالور اینستاگرام به صورت API (Python, Fast API, requests)",\n      "Pybsky: Python BlueSky client (Python, requests)",\n     "سیستم No Code AI برای سیاستگذاری (پروژه بنیاد ملی نخبگان) (Python, FastAPI, Scikit-learn, tensorflow)",\n     "تدریس دوره جامع هوش مصنوعی (Python, Numpy, Pandas, Matplotlib, Scikit-learn, Tensorflow)"\n  ],\n  "publications": [\n    "آموزش وب‌کاوی با scrapy",\n    "آموزش زبان برنامه‌نویسی Ruby"\n  ]\n}\n```',
        #     ],
        # },
        {
            "role": "user",
            "parts": [
                "all the information with link must be linked",
            ],
        },
        # {
        #     "role": "model",
        #     "parts": [
        #         '```json\n{\n  "bio": "متخصص با تجربه در زمینه توسعه نرم افزار و تحلیل داده، با تمرکز ویژه بر کاربرد هوش مصنوعی و یادگیری ماشین در تحلیل داده های حجیم. دارای سابقه موفق در استخراج بینش های ارزشمند از داده ها، مدل سازی و ارائه گزارش های مدیریتی.",\n  "experiences": [\n    "مهندس نرم افزار در سازمان بهداشت و درمان صنعت نفت (1396-1397): توسعه سیستم های نرم افزاری با استفاده از WCF و ASP.NET MVC.",\n    "توسعه دهنده Python در سپهر پیام تدبیر ایرانیان (1396-1397): توسعه سیستم های جمع آوری داده با استفاده از Python، Scrapy و Crawl برای شبکه های اجتماعی.",\n    "توسعه دهنده وب Backend در سپهر پیام تدبیر ایرانیان (1397-1398): توسعه سیستم های وب با استفاده از PHP، CodeIgniter و MySQL.",\n    "توسعه دهنده وب Backend در موسسه رشد و سرمایه انسانی ایرانیان (رسا) (1398-1400): توسعه API های RESTful با استفاده از PHP، Laravel و MySQL.",\n    "توسعه دهنده وب Backend در سامانه خرید آنلاین بیمه (آلبیم) (1400-1401): توسعه سیستم های وب با استفاده از .net core5، MSSQL Server و React Js.",\n     "توسعه دهنده وب Backend در دانش گستر سینا (1402): توسعه سیستم های داده با استفاده از Python, FastAPI, Celery, requests, scrapy.",\n    "مدرس زبان‌های برنامه نویسی و هوش مصنوعی در مجتمع فنی نیاوران (1399-تاکنون): تدریس Python، Data Analysis، Machine Learning و Deep Learning.",\n    "طراح سامانه‌های هوشمند مدیریت سرمایه انسانی در پژوهشکده سرمایه انسانی دافوس آجا (1402-1403): توسعه سیستم های هوشمند با استفاده از Python، Tensorflow و Pytorch."\n  ],\n  "skills": [\n    "مسلط به زبان برنامه نویسی Python",\n    "مسلط به کتابخانه های پایتون برای تحلیل داده و یادگیری ماشین (Numpy, Pandas, Scikit-learn, Tensorflow, Pytorch)",\n    "تجربه در توسعه API های RESTful با استفاده از FastAPI و Laravel",\n    "تجربه در کار با پایگاه داده های SQL و NoSQL (MySQL, SQL Server, NEO4J)",\n    "تجربه در جمع آوری داده از وب با استفاده از Scrapy و Selenium",\n    "تجربه در پیاده سازی مدل های یادگیری ماشین و هوش مصنوعی",\n    "تجربه در ارائه گزارش های مدیریتی و تحلیل داده ها"\n  ],\n  "courses": [\n      "Network+",\n      "CCNA",\n      "MCSA 2012",\n      "SPSS",\n      "SQL Server",\n      "Web Application Vulnerabilities and Security Attacks",\n      "Modern PHP"\n  ],\n    "projects": [\n    "پیاده‌سازی وب سرویس از روی مدل برداری ساخته شده از کلمات ویکی‌پدیا فارسی با الگوریتم word2vec (Python, Word Embedding , Flask, Word2Vec)",\n    "سیستم مدیریت قرارداد پزشکان سازمان بهداشت و درمان صنعت نفت (WCF, EntityFramework Code First)",\n     "سیستم رصد و تحلیل اخبار فرهنگی: کراول و پردازش محتوای وبسایت های خبری (PHP, CodeIgniter, Python, NLP)",\n     "کراولر جابینجا بهمراه بات تلگرامی دریافت آگهی‌ها (Python, Telegram Bot API) [لینک](https://github.com/softrebel/jobOnjaBot)",\n     "سیستم بات اینستاگرام، پیدا کردن id از روی شماره افراد، مدیریت اتوماتیک فالو و آنفالو (Python, API Instagram) [لینک](https://github.com/softrebel/instagram_private_api)",\n     "سیستم کراول maps google :جمع آوری نقاط مکانی بهمراه مشخصات (Python, Selenium, BeautifulSoup) [لینک](https://github.com/softrebel/googleMapCrawler)",\n     "سیستم کانون ارزیابی کارکنان (Assessment Center) (PHP, Laravel web API, vue.js) [لینک](https://ac.rasa-hr.com/)",\n     "سیستم آزمون/ گزینش آنلاین رسا (PHP, Laravel web API, vue.js) [لینک](https://ac.rasa-hr.com/)",\n     "کراول فیلم‌های ایرانی دهه 90 فیلیمو و ذخیره و بازیابی در دیتابیس گرافی NEO4J (Python, NEO4J Graph Database)",\n     "سیستم مشاوره و خرید بیمه آنلاین (.net core 5 web API, react.js) [لینک](https://allbim.org)",\n     "استخراج داده از سایت ایرانداک (Python, scrapy) [لینک](https://github.com/softrebel/irandoc-scraper)",\n     "تحلیل شبکه ارتباطی مقالات ایرانداک (Python, igraph) [لینک](https://github.com/softrebel/irandoc_crawler)",\n     "MapReduce sample project using mrjob (Python, mrjob) [لینک](https://github.com/softrebel/mrjob_exercise)",\n     "PySpark Machine Learning Project (Python, pyspark) [لینک](https://github.com/softrebel/spark_project)",\n     "A Comparison of Centrality Measures for Community Evolution Discovery (Python, igraph, GED) [لینک](https://doi.org/10.1109/ICWR54782.2022.9786228)",\n     "وب‌سرویس لاگین و دریافت فالور اینستاگرام به صورت API (Python, Fast API, requests) [لینک](https://github.com/softrebel/fastapi-instagram-api)",\n      "Pybsky: Python BlueSky client (Python, requests) [لینک](https://github.com/softrebel/pybsky)",\n     "سیستم No Code AI برای سیاستگذاری (پروژه بنیاد ملی نخبگان) (Python, FastAPI, Scikit-learn, tensorflow)",\n     "تدریس دوره جامع هوش مصنوعی (Python, Numpy, Pandas, Matplotlib, Scikit-learn, Tensorflow) [لینک](https://mftniavaran.com/class/ai/)"\n  ],\n  "publications": [\n    "آموزش وب‌کاوی با scrapy [لینک](https://www.digikala.com/product/dkp-10782989/)",\n    "آموزش زبان برنامه‌نویسی Ruby [لینک](https://www.digikala.com/product/dkp-12308758)"\n  ]\n}\n```',
        #     ],
        # },
        # {
        #     "role": "user",
        #     "parts": [
        #         "اگه دنبال چالش های مدیریت میلیاردها رکورد اطلاعاتی تو صنایع مخابراتی و بانکی هستید و دوست دارید دانش حوزه مهندسی داده رو کسب کنید، رزومه تون رو برای ما بفرستید.دانش، مهارت و توانایی‌های مورد نیاز:تسلط کافی به زبان انگلیسی جهت R&Dآشنایی با  Python : متوسط به بالاآشنایی با Java: ابتدایی به بالاآشنایی با زبان‌  SQL جهت پردازش‌های داده: متوسط به بالاآشنایی با پیاده‌سازی و مدیریت یکی از پایگاه‌های داده.RDBMS یا NoSqlآشنایی با دستورات پایه ای  لینوکس.آشنایی با Git مزیت محصوب میشود.آشنایی با Docker مزیت محصوب میشود.آشنایی با ابزارهای ETL مزیت محصوب میشود. شرح شغل : طراحی و پیاده‌سازی فرایندهای تحلیل داده و بهبود عملکرد سیستم‌های مهندسی داده.طراحی و مدلینگ داده بر اساس نیازهای جدید.پیاده‌سازی و نگهداری پایپ‌لاین‌های داده مقیاس‌پذیر.مانیتورینگ سرویس‌های داده ایی و حل مسائل در صورت خرابی.ویژگی های فردی مورد انتظار:یادگیری و درک مطلب بالادقت در انجام وظایف محوله خلاق  در پیاده سازی و طراحیاخلاق و رفتار حرفه ای در تعاملات سازمانیمشارکت و انعطاف پذیری در کار تیمیاجتماعی و توانمند در ارتباطات مزایای دوره کارآموزی :آشنایی با مفاهیم دیریت کلان داده آشنایی با اکوسیستم big dataتجربه در طراحی و ساخت سیستم‌های پردازش داده و ابزارهای مرتبط.آشنایی با ابزارهای مختلف ذخیره سازی داده و روش های مرتبط.RDBMS & NoSQLآشنای با ابزارهای مدیریت داده .تجربه ی عملی کدنویسی در محیط توزیع شده.آشنایی با ابزارهای مدیریت جریان داده .آشنایی با ابزارهای مدیریت جریان کار.تعریف دوره:مدت دوره کارآموزی سه ماه میباشد.امکان استخدام بعد از دوره و آزمون.مزایای همکاری با سان: کار در محیطی دوستانه و پویا کار همراه با یادگیری و ارتقای شغلی کار در شرکتی با سابقه با پروژ‌ه‌های فعال در سازمان های بزرگ بیمه تکمیلی SOS بیمه عمر و حوادثکافه و فضای بازی",
        #     ],
        # },
        # {
        #     "role": "model",
        #     "parts": [
        #         '```json\n{\n  "bio": "متخصص با تجربه در زمینه توسعه نرم افزار و تحلیل داده، با تمرکز ویژه بر مهندسی داده و پردازش داده های حجیم. دارای سابقه موفق در طراحی و پیاده سازی سیستم های داده مقیاس پذیر، استخراج بینش های ارزشمند از داده ها و ارائه گزارش های مدیریتی. علاقه مند به یادگیری و کسب دانش در حوزه های کلان داده و سیستم های توزیع شده.",\n  "experiences": [\n    "مهندس نرم افزار در سازمان بهداشت و درمان صنعت نفت (1396-1397): توسعه سیستم های نرم افزاری با استفاده از WCF و ASP.NET MVC.",\n    "توسعه دهنده Python در سپهر پیام تدبیر ایرانیان (1396-1397): توسعه سیستم های جمع آوری داده با استفاده از Python، Scrapy و Crawl برای شبکه های اجتماعی.",\n    "توسعه دهنده وب Backend در سپهر پیام تدبیر ایرانیان (1397-1398): توسعه سیستم های وب با استفاده از PHP، CodeIgniter و MySQL.",\n    "توسعه دهنده وب Backend در موسسه رشد و سرمایه انسانی ایرانیان (رسا) (1398-1400): توسعه API های RESTful با استفاده از PHP، Laravel و MySQL.",\n    "توسعه دهنده وب Backend در سامانه خرید آنلاین بیمه (آلبیم) (1400-1401): توسعه سیستم های وب با استفاده از .net core5، MSSQL Server و React Js.",\n     "توسعه دهنده وب Backend در دانش گستر سینا (1402): توسعه سیستم های داده با استفاده از Python, FastAPI, Celery, requests, scrapy.",\n    "مدرس زبان‌های برنامه نویسی و هوش مصنوعی در مجتمع فنی نیاوران (1399-تاکنون): تدریس Python، Data Analysis، Machine Learning و Deep Learning.",\n     "طراح سامانه‌های هوشمند مدیریت سرمایه انسانی در پژوهشکده سرمایه انسانی دافوس آجا (1402-1403): توسعه سیستم های هوشمند با استفاده از Python، Tensorflow و Pytorch."\n  ],\n  "skills": [\n    "مسلط به زبان برنامه نویسی Python",\n    "آشنایی متوسط به بالا با زبان SQL جهت پردازش داده",\n     "آشنایی با زبان Java",\n    "مسلط به کتابخانه های پایتون برای تحلیل داده و یادگیری ماشین (Numpy, Pandas, Scikit-learn, Tensorflow, Pytorch)",\n    "تجربه در کار با پایگاه داده های SQL و NoSQL (MySQL, SQL Server, NEO4J)",\n    "تجربه در پیاده سازی و مدیریت پایگاه داده های RDBMS و NoSQL",\n    "تجربه در جمع آوری داده از وب با استفاده از Scrapy و Selenium",\n    "آشنایی با ابزارهای ETL",\n    "آشنایی با دستورات پایه ای لینوکس",\n    "آشنایی با Git",\n    "آشنایی با Docker",\n     "تجربه در پیاده سازی مدل های یادگیری ماشین و هوش مصنوعی",\n    "تجربه در ارائه گزارش های مدیریتی و تحلیل داده ها"\n  ],\n  "courses": [\n      "Network+",\n      "CCNA",\n      "MCSA 2012",\n      "SPSS",\n      "SQL Server",\n      "Web Application Vulnerabilities and Security Attacks",\n      "Modern PHP"\n  ],\n    "projects": [\n    "پیاده‌سازی وب سرویس از روی مدل برداری ساخته شده از کلمات ویکی‌پدیا فارسی با الگوریتم word2vec (Python, Word Embedding , Flask, Word2Vec)",\n    "سیستم مدیریت قرارداد پزشکان سازمان بهداشت و درمان صنعت نفت (WCF, EntityFramework Code First)",\n     "سیستم رصد و تحلیل اخبار فرهنگی: کراول و پردازش محتوای وبسایت های خبری (PHP, CodeIgniter, Python, NLP)",\n     "کراولر جابینجا بهمراه بات تلگرامی دریافت آگهی‌ها (Python, Telegram Bot API) [لینک](https://github.com/softrebel/jobOnjaBot)",\n     "سیستم بات اینستاگرام، پیدا کردن id از روی شماره افراد، مدیریت اتوماتیک فالو و آنفالو (Python, API Instagram) [لینک](https://github.com/softrebel/instagram_private_api)",\n     "سیستم کراول maps google :جمع آوری نقاط مکانی بهمراه مشخصات (Python, Selenium, BeautifulSoup) [لینک](https://github.com/softrebel/googleMapCrawler)",\n     "سیستم کانون ارزیابی کارکنان (Assessment Center) (PHP, Laravel web API, vue.js) [لینک](https://ac.rasa-hr.com/)",\n     "سیستم آزمون/ گزینش آنلاین رسا (PHP, Laravel web API, vue.js) [لینک](https://ac.rasa-hr.com/)",\n     "کراول فیلم‌های ایرانی دهه 90 فیلیمو و ذخیره و بازیابی در دیتابیس گرافی NEO4J (Python, NEO4J Graph Database)",\n     "سیستم مشاوره و خرید بیمه آنلاین (.net core 5 web API, react.js) [لینک](https://allbim.org)",\n     "استخراج داده از سایت ایرانداک (Python, scrapy) [لینک](https://github.com/softrebel/irandoc-scraper)",\n     "تحلیل شبکه ارتباطی مقالات ایرانداک (Python, igraph) [لینک](https://github.com/softrebel/irandoc_crawler)",\n     "MapReduce sample project using mrjob (Python, mrjob) [لینک](https://github.com/softrebel/mrjob_exercise)",\n     "PySpark Machine Learning Project (Python, pyspark) [لینک](https://github.com/softrebel/spark_project)",\n     "A Comparison of Centrality Measures for Community Evolution Discovery (Python, igraph, GED) [لینک](https://doi.org/10.1109/ICWR54782.2022.9786228)",\n     "وب‌سرویس لاگین و دریافت فالور اینستاگرام به صورت API (Python, Fast API, requests) [لینک](https://github.com/softrebel/fastapi-instagram-api)",\n      "Pybsky: Python BlueSky client (Python, requests) [لینک](https://github.com/softrebel/pybsky)",\n     "سیستم No Code AI برای سیاستگذاری (پروژه بنیاد ملی نخبگان) (Python, FastAPI, Scikit-learn, tensorflow)",\n     "تدریس دوره جامع هوش مصنوعی (Python, Numpy, Pandas, Matplotlib, Scikit-learn, Tensorflow) [لینک](https://mftniavaran.com/class/ai/)"\n  ],\n  "publications": [\n    "آموزش وب‌کاوی با scrapy [لینک](https://www.digikala.com/product/dkp-10782989/)",\n    "آموزش زبان برنامه‌نویسی Ruby [لینک](https://www.digikala.com/product/dkp-12308758)"\n  ]\n}\n```',
        #     ],
        # },
        {
            "role": "user",
            "parts": [
                "in experiences section, every head must be bold and description is normal"
            ],
        },
        {
            "role": "user",
            "parts": [
                "consider that educations is mandatory section, all educations must be included all the time and every head must be bold and description is normal"
            ],
        },
    ]

    gemini_handler = GeminiHandler(
        model_name="gemini-2.0-flash-exp",
        api_key=project_configs.GEMINI_API_KEY,
        generation_config=default_generation_config,
        chat_history=chat_history,
    )
    return gemini_handler


def generate_resume_for_job(
    llm_handler: BaseLLMHandler, job_description: str, base_resume: ResumeBase
) -> ResumeGenerated:
    response_json = llm_handler.send_prompt(job_description)
    response = json.loads(response_json)
    return ResumeGenerated(
        **response,
        title=base_resume.title,
    )


def list_to_md(items: list[str]) -> str:
    return "\n".join([f"- {x}" for x in items])


def md_to_html(md: str) -> str:
    return markdown.markdown(md)


def save_pdf(html: str, file_path: str) -> None:
    options = {
        "page-size": "Letter",
        "encoding": "UTF-8",
    }

    output_path = file_path

    config = pdfkit.configuration(wkhtmltopdf=project_configs.WKHTMLTOPDF_PATH)
    pdfkit.from_string(
        f"{project_configs.TEMPLATE_HEADER}{html}",
        output_path,
        configuration=config,
        options=options,
    )
