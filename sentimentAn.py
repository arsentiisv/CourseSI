import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from translate import Translator


class Sentiment:

    def vrem1(self, s):
        s = s.split('-')
        s = str(s[2] + '.' + s[1] + '.' + s[0])
        return s

    def vrem2(self, s):
        s = s.split('.')
        s = str(s[2] + '-' + s[1] + '-' + s[0])
        return s

    def obed(self, el, eli, ele):
        bta = pd.concat([el, eli, ele])
        bta = bta.reset_index(drop=True)
        return bta

    def repka(self, s):
        s = s[4:-5]
        s = s.replace('</span>', '')
        s = s.replace('<span class="allocation_found">', '')
        return s

    def fim(self, s):
        s = s[s.find('>') + 1:-4]
        return s

    def LentaNews(self, zapros, start, finish):
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ru,en;q=0.9',
            'Connection': 'keep-alive',
            # 'Cookie': 'lid=vAsAAOHiXWcmQGRmAWS8AQB=; tmr_lvid=e1c03db4d52cb1d5d6728931f33ce66e; tmr_lvidTS=1734206178094; _ym_uid=1734206178829234461; _ym_d=1734206178; adtech_uid=219b07cf-bb76-43af-8c21-df8c2f459068%3Alenta.ru; top100_id=t1.80674.886928341.1734206178198; __ldr_auto_key=dc3bf7eb-b716-4760-80b0-acbf8a67ace3; VARIANT=0; chash=gnELGAra4z; vpuid=1734361808.311-1843067462000294; _ym_isad=2; domain_sid=NFrMg_-1JkIfFBa5MnHA-%3A1734603690087; t3_sid_4422985=s1.1847186340.1734606301805.1734607985403.2.26; t3_sid_7643964=s1.1962430312.1734607890200.1734607985405.2.28; t3_sid_7356279=s1.1167604220.1734607890208.1734607985406.2.26; lids=482540130A7DDFB5; tmr_detect=0%7C1734617599702; t3_sid_80674=s1.1170661902.1734617583945.1734617619365.6.27',
            'Referer': 'https://lenta.ru/search?query=%D0%A1%D0%B1%D0%B5%D1%80%D0%B1%D0%B0%D0%BD%D0%BA',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36',
            'sec-ch-ua': '"Chromium";v="130", "YaBrowser";v="24.12", "Not?A_Brand";v="99", "Yowser";v="2.5"',
            'sec-ch-ua-mobile': '?1',
            'sec-ch-ua-platform': '"Android"',
        }
        params = {
            'query': f'{zapros}',
            'from': '0',
            'size': '10',
            'sort': '2',
            'title_only': '0',
            'domain': '1',
            'modified,format': 'yyyy-MM-dd',
            'modified,from': f'{start}',
            'modified,to': f'{finish}',
        }
        response = requests.get('https://lenta.ru/search/v2/process', params=params, headers=headers)
        vr = response.json()['total_found']
        params['size'] = vr
        response = requests.get('https://lenta.ru/search/v2/process', params=params, headers=headers)
        da = response.json()
        del vr
        el = pd.DataFrame(da['matches'])[['title', 'pubdate']]
        el['pubdate'] = el['pubdate'].apply(lambda x: datetime.fromtimestamp(x))
        el['time'] = el['pubdate'].apply(lambda x: str(x)[11:])
        el['pubdate'] = el['pubdate'].apply(lambda x: str(x)[:10])
        el.rename(columns={'pubdate': 'date'}, inplace=True)
        el['source'] = 'Lenta'
        return el

    def RbkNews(self, zapros, start, finish):
        start = self.vrem1(start)
        finish = self.vrem1(finish)
        headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ru,en;q=0.9',
            'Connection': 'keep-alive',
            # 'Cookie': 'foxTailParam_sticky_bottom_inner_mobile_is_hidden=true; _ym_uid=1729541611621080959; _ym_d=1729541611; _ga_BT8R8SQ8WR=GS1.2.1729541611.1.0.1729541611.60.0.0; _fbp=fb.1.1729541611746.633953009141244780; splituid=uUjlWGczHj9847KxA3CGAg==; tmr_lvid=a2cc9d63f482add670b7b6ed5f8f4763; tmr_lvidTS=1731403329405; __rmid=fw2GLSkbRR6E8WecHC6ThQ; popmechanic_sbjs_migrations=popmechanic_1418474375998%3D1%7C%7C%7C1471519752600%3D1%7C%7C%7C1471519752605%3D1; livetv-state=off; _ga=GA1.1.1218022124.1729541611; _ga_4M539MHND5=GS1.1.1732479954.1.1.1732480051.28.0.0; admitad_uid=; _ym_isad=2; domain_sid=3T_uI-T9wPA6Z4GjBAqy3%3A1734603458305; js_d=false; mindboxDeviceUUID=36ba830c-9de5-41e3-a3ff-b1e06c9fefd9; directCrm-session=%7B%22deviceGuid%22%3A%2236ba830c-9de5-41e3-a3ff-b1e06c9fefd9%22%7D; qrator_msid2=v2.0.1734616875.872.d42e0a635PWBJ4ZM|ZLtFhXgf0DtkX1CY|KOvO7IakFilaOAcKkZ1Jan1Yg8q/rbAfW+mDQbj1xDSB6gopl2ScJxMmZgmFubB2EyUw2WaPDyFkVgDztGla2Q==-tOFvce254imMasfA5HHZczHrjBE=; __rmsid=8jcBWNfDTPeuZvgF24ulbg; _ym_visorc=b; tmr_detect=0%7C1734617223060',
            'Referer': 'https://www.rbc.ru/search/?dateTo=19.12.2024&dateFrom=19.11.2024&query=%D0%9C%D0%B0%D0%BA%D0%B3%D1%80%D0%B5%D0%B3%D0%BE%D1%80',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'sec-ch-ua': '"Chromium";v="130", "YaBrowser";v="24.12", "Not?A_Brand";v="99", "Yowser";v="2.5"',
            'sec-ch-ua-mobile': '?1',
            'sec-ch-ua-platform': '"Android"',
        }
        params = {
            'dateTo': f'{finish}',
            'dateFrom': f'{start}',
            'query': f'{zapros}',
            'page': '0',
        }
        da = {'items': []}
        nst = 0
        while True:
            params['page'] = nst
            response = requests.get('https://www.rbc.ru/search/ajax/', params=params, headers=headers)
            vr = response.json()
            for i in range(len(vr['items'])):
                if (zapros in (vr['items'][i]['title'].split()) or (
                        vr['items'][i]['body'] != None and zapros in (vr['items'][i]['body'].split()))):
                    da['items'].append(vr['items'][i])
            if not vr['moreExists']:
                break
            if vr['moreExists']:
                nst += 1
        eli = pd.DataFrame(da['items'])[['title', 'publish_date_t']]
        # 'project', 'category']]
        eli['publish_date_t'] = eli['publish_date_t'].apply(lambda x: datetime.fromtimestamp(x))
        eli['time'] = eli['publish_date_t'].apply(lambda x: str(x)[11:])
        eli['publish_date_t'] = eli['publish_date_t'].apply(lambda x: str(x)[:10])
        eli.rename(columns={'publish_date_t': 'date'}, inplace=True)
        eli['source'] = 'RBK'
        return eli

    ###  Эту функцию не используем, но она работает
    def RiaNews(self, zapros, start, finish):
        headers = {
            'accept': 'text/html, */*; q=0.01',
            'accept-language': 'ru,en;q=0.9',
            # 'cookie': '_ym_uid=1728823092934280905; _ym_d=1728823092; tmr_lvid=aefb61c0fff47f60272e3638c429f576; tmr_lvidTS=1728823091844; _ymab_param=mNAVXTq6ahOmxROABS3Vg5QkwkI6wVPa-D5_r7OqVsCEyoA1k20dC4hMnFYXa2mr7xrd4TjizkxEJ8tL6qvvHG6FeoM; _ga=GA1.1.2106071890.1728823096; riaru=6749a1b7d79d1d851c2917c7; ab_amb2_smrz=a; fb_check5=1; _ym_isad=2; domain_sid=8XqyCrWNojcddViK5WxA8%3A1735467438206; _ga_0MXQ5FCEG3=GS1.1.1735467437.4.1.1735467447.0.0.0; _pk_ref.ria.ddd3=%5B%22%22%2C%22%22%2C1735491895%2C%22https%3A%2F%2Fyandex.ru%2F%22%5D; _ym_visorc=b; _pk_id.ria.ddd3=0842a201f2e76f64.1728823092.11.1735493123.1735488229.; _pk_ses.ria.ddd3=%7B%22depth%22%3A14%7D; ispwa-user-visits=73; tmr_detect=0%7C1735493125950',
            'priority': 'u=1, i',
            'referer': 'https://ria.ru/search/?query=%D1%81%D0%B1%D0%B5%D1%80%D0%B1%D0%B0%D0%BD%D0%BA',
            'sec-ch-ua': '"Chromium";v="130", "YaBrowser";v="24.12", "Not?A_Brand";v="99", "Yowser";v="2.5"',
            'sec-ch-ua-mobile': '?1',
            'sec-ch-ua-platform': '"Android"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36',
            'x-requested-with': 'XMLHttpRequest',
        }
        q = 0
        da = {'title': [], 'date': [], 'time': []}
        params = {'query': f'{zapros}',
                  'offset': f'{q}',
                  'date_from': f'{start}',
                  'date_to': f'{finish}',
                  'sort[]': 'date'}
        session = requests.session()
        session.headers.update(headers)
        response = session.get('https://ria.ru/services/search/getmore/', params=params)
        soup = BeautifulSoup(response.text, features="lxml")
        ch = soup.find('div', class_='list-items-loaded')['data-count']
        cyc = int(int(ch) / 20) + 1
        while cyc > 0:
            response = session.get(f'https://ria.ru/services/search/getmore/', params=params)
            soup = BeautifulSoup(response.text, features="lxml")
            block = soup.findAll(class_='list-item__title color-font-hover-only')
            for item in block:
                vrreq = requests.get(item['href'])
                sup = BeautifulSoup(vrreq.text, features='lxml')
                tex = sup.find(class_='article__title').text
                if zapros.lower() in str(tex).lower():
                    da['title'].append(tex)
                    vrem = sup.find(class_='article__info-date').text.split()
                    da['date'].append(vrem[1])
                    da['time'].append(vrem[0])
            q = q + 20
            cyc = cyc - 1
            params['offset'] = q
        ele = pd.DataFrame(da)
        ele['source'] = 'Ria'
        ele['date'] = ele['date'].apply(lambda x: self.vrem2(x))
        ele['time'] = ele['time'].apply(lambda x: x + ':00')
        return ele

    def AifNews(self, zapros, nachalo, konchalo):
        headers = {
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'accept-language': 'ru,en;q=0.9',
            'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            # 'cookie': '__ddg1_=omGy4UMEaRpefOuv2ksL; _ym_uid=1734206063255805456; _ym_d=1734206063; tmr_lvid=f08117dc1040ab4209bc7098f3ebccd3; tmr_lvidTS=1734206063100; _ga=GA1.1.658255408.1734206064; subscription_popup_min_state=week_view1; __ddg9_=94.233.241.249; aif_sid=ef2325c25337de9e6f764915c283fc63; CookieMessenger=; betweenx=1; subscription_popup_state=%7B%22key%22%3A%22week_view2%22%2C%22timeout%22%3A0%7D; domain_sid=qoAwdTIRH-DLhilHsY_ar%3A1735897289458; _ym_isad=2; _ym_visorc=b; fid=7e0f49c2-29fb-40e8-aa23-40adab1f321f; __upin=qvHEqGpH3cjUGD6Cb5UUxQ; _ac_cid=0300007FC9B077679622752702C34E88; _ac_oid=50af1215806c32aaaf9f88101f1b72d7%3A1735900891350; tmr_detect=0%7C1735897292157; ma_vis_id_last_sync_3485699018=1735897292531; ma_prevVisId_3485699018=83bad936ccf9e8987e2e6313e6c981c7; ma_id=5494524361735897291226; _ga_RFZR8VEEZD=GS1.1.1735897289.4.1.1735897302.47.0.0; __ddg8_=Ebv5WJ1H2ECNFKC8; __ddg10_=1735897288',
            'origin': 'https://aif.ru',
            'priority': 'u=1, i',
            'sec-ch-ua': '"Chromium";v="130", "YaBrowser";v="24.12", "Not?A_Brand";v="99", "Yowser";v="2.5"',
            'sec-ch-ua-mobile': '?1',
            'sec-ch-ua-platform': '"Android"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36',
            'x-requested-with': 'XMLHttpRequest',
        }
        q = 1
        da = {'title': [],
              'date': [],
              # 'type': [],
              }
        while True:
            data = {
                'page': f'{q}',
            }
            response = requests.post(
                f'https://aif.ru/search/index/index/content_type/2/from/{nachalo}/to/{konchalo}/text/{zapros}',
                headers=headers, data=data)
            vr = response.json()
            sta = vr['isFinished']
            soup = BeautifulSoup(vr['data'], features="lxml")
            block = soup.findAll(class_='list_item')
            for item in block:
                da['title'].append(self.repka(str(item.find('h3'))))
                da['date'].append(str(item.find(class_='text_box__date'))[29:-7])
                # da['type'].append(fim(str(item.find(class_="rubric_link no_title_element_js"))))
            if not sta:
                q += 1
            if sta:
                break
        elo = pd.DataFrame(da)
        elo['time'] = elo['date'].apply(lambda x: x[11:] + ':00')
        elo['date'] = elo['date'].apply(lambda x: self.vrem2(x[:10]))
        elo['source'] = 'AiF'
        return elo

    def simS(self, str1, str2, zn):
        len_str1 = len(str1)
        len_str2 = len(str2)
        str1 = str1.lower()
        str2 = str2.lower()
        dp = [[0] * (len_str2 + 1) for i in range(len_str1 + 1)]
        for i in range(len_str1 + 1):
            dp[i][0] = i
        for j in range(len_str2 + 1):
            dp[0][j] = j
        for i in range(1, len_str1 + 1):
            for j in range(1, len_str2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,
                               dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + cost)
        if dp[len_str1][len_str2] <= zn:
            return 1
        return 0

### Кэф прикручивается для более точного поиска, больше -> точнее
    def Nstr(self, str1, spis):
        kef = 1.5
        for el in spis.split():
            if self.simS(str1, el, max(len(str1), len(el)) // kef):
                return 1
        return 0

    def tt(self, text):
        translator = Translator(from_lang='ru', to_lang='en')
        try:
            translated_text = translator.translate(text)
            return translated_text
        except Exception as e:
            return f"Error: {e}"

    def sentAn(self, s):
        analyzer = SentimentIntensityAnalyzer()
        rev = analyzer.polarity_scores(s)
        return rev['compound']

### Здесь уже объединение всех функций
    def SAn(self, z, n, k):
        da_lenta = self.LentaNews(z, n, k)
        da_rbk = self.RbkNews(z, n, k)
        da_aif = self.AifNews(z, n, k)
        da = self.obed(da_lenta, da_rbk, da_aif)
### Фильтрация, где есть что-то похожее на запрос в заголовке
        da = da[da['title'].apply(lambda x: self.Nstr(z, x)) == 1]
### Единственная проблема, что сентимент анализ через nltk принимает только английский текст, а новости сами на русском
        # da['perevod']=da['title'].apply(lambda x: self.tt(x))
        # da['sent']=da['perevod'].apply(lambda x: self.sentAn(x))
        return da

zap = 'Сбербанк'
na = '2025-01-01'
kon = '2025-05-01'
senty = Sentiment()
tabl=senty.SAn(zap,na,kon)