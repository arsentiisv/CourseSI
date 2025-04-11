import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup



def vrem1(s):
    s = s.split('-')
    s = str(s[2] + '.' + s[1] + '.' + s[0])
    return s


def vrem2(s):
    s = s.split('.')
    s = str(s[2] + '-' + s[1] + '-' + s[0])
    return s


def obed(el, eli, ele):
    bta = pd.concat([el, eli, ele])
    # bta['sent']=bta['title'].apply(lambda x: sentAn(tt(x)))
    return bta


# def repka(s):
#     s = s[4:-5]
#     s = s.replace('</span>', '')
#     s = s.replace('<span class="allocation_found">', '')
#     return s


# def fim(s):
#     s = s[s.find('>') + 1:-4]
#     return s


def LentaNews(zapros, start, finish):
    cookies = {
        'lid': 'vAsAAOHiXWcmQGRmAWS8AQB=',
        'tmr_lvid': 'e1c03db4d52cb1d5d6728931f33ce66e',
        'tmr_lvidTS': '1734206178094',
        '_ym_uid': '1734206178829234461',
        '_ym_d': '1734206178',
        'adtech_uid': '219b07cf-bb76-43af-8c21-df8c2f459068%3Alenta.ru',
        'top100_id': 't1.80674.886928341.1734206178198',
        '__ldr_auto_key': 'dc3bf7eb-b716-4760-80b0-acbf8a67ace3',
        'VARIANT': '0',
        'chash': 'gnELGAra4z',
        'vpuid': '1734361808.311-1843067462000294',
        't3_sid_4422985': 's1.1847186340.1734606301805.1734607985403.2.26',
        't3_sid_7643964': 's1.1962430312.1734607890200.1734607985405.2.28',
        't3_sid_7356279': 's1.1167604220.1734607890208.1734607985406.2.26',
        'lids': '482056BF349E841E',
        '_ym_isad': '2',
        'domain_sid': 'NFrMg_-1JkIfFBa5MnHA-%3A1738592897683',
        'tmr_detect': '0%7C1738592909690',
        't3_sid_80674': 's1.1153266142.1738592897404.1738592949101.8.22',
    }

    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ru,en;q=0.9',
        'Connection': 'keep-alive',
        # 'Cookie': 'lid=vAsAAOHiXWcmQGRmAWS8AQB=; tmr_lvid=e1c03db4d52cb1d5d6728931f33ce66e; tmr_lvidTS=1734206178094; _ym_uid=1734206178829234461; _ym_d=1734206178; adtech_uid=219b07cf-bb76-43af-8c21-df8c2f459068%3Alenta.ru; top100_id=t1.80674.886928341.1734206178198; __ldr_auto_key=dc3bf7eb-b716-4760-80b0-acbf8a67ace3; VARIANT=0; chash=gnELGAra4z; vpuid=1734361808.311-1843067462000294; t3_sid_4422985=s1.1847186340.1734606301805.1734607985403.2.26; t3_sid_7643964=s1.1962430312.1734607890200.1734607985405.2.28; t3_sid_7356279=s1.1167604220.1734607890208.1734607985406.2.26; lids=482056BF349E841E; _ym_isad=2; domain_sid=NFrMg_-1JkIfFBa5MnHA-%3A1738592897683; tmr_detect=0%7C1738592909690; t3_sid_80674=s1.1153266142.1738592897404.1738592949101.8.22',
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
        'type':'1',
        'modified,from': f'{start}',
        'modified,to': f'{finish}',
    }
    response = requests.get('https://lenta.ru/search/v2/process', params=params,headers=headers)
    vr = response.json()['total_found']
    params['size'] = vr
    response = requests.get('https://lenta.ru/search/v2/process', params=params,headers=headers)
    da = response.json()
    del vr
    el = pd.DataFrame(da['matches'])[['title', 'pubdate']]
    el['pubdate'] = el['pubdate'].apply(lambda x: datetime.fromtimestamp(x))
    el['time'] = el['pubdate'].apply(lambda x: str(x)[11:])
    el['pubdate'] = el['pubdate'].apply(lambda x: str(x)[:10])
    el.rename(columns={'pubdate': 'date'}, inplace=True)
    el['source'] = 'Lenta'
    return el


def RbkNews(zapros, start, finish):
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'ru,en;q=0.9',
        'Connection': 'keep-alive',
        # 'Cookie': '_ym_uid=1729541611621080959; _ym_d=1729541611; _ga_BT8R8SQ8WR=GS1.2.1729541611.1.0.1729541611.60.0.0; splituid=uUjlWGczHj9847KxA3CGAg==; tmr_lvid=a2cc9d63f482add670b7b6ed5f8f4763; tmr_lvidTS=1731403329405; __rmid=fw2GLSkbRR6E8WecHC6ThQ; popmechanic_sbjs_migrations=popmechanic_1418474375998%3D1%7C%7C%7C1471519752600%3D1%7C%7C%7C1471519752605%3D1; livetv-state=off; _ga=GA1.1.1218022124.1729541611; _ga_4M539MHND5=GS1.1.1732479954.1.1.1732480051.28.0.0; mindboxDeviceUUID=36ba830c-9de5-41e3-a3ff-b1e06c9fefd9; directCrm-session=%7B%22deviceGuid%22%3A%2236ba830c-9de5-41e3-a3ff-b1e06c9fefd9%22%7D; toprbc_region=world; toprbc_date=Thu%20Jan%2016%202025%2000%3A00%3A00%20GMT%2B0300%20(%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B0%2C%20%D1%81%D1%82%D0%B0%D0%BD%D0%B4%D0%B0%D1%80%D1%82%D0%BD%D0%BE%D0%B5%20%D0%B2%D1%80%D0%B5%D0%BC%D1%8F); admitad_uid=; ad_user_gender_new=2; ad_user_age=27; js_d=false; qrator_msid2=v2.0.1738593701.235.59af12bdeiYXdMIX|dzDyoJnwgHvH04CY|JwairmZL2AL+6kec75EVUCXCeDpRgSDfQN0xzh/XDVFYe8+rbBPPxSjzHrVug4QH02AMcKy2qB1XyhoAyrtRRA==-JgymbbfeyHzO6QAptl8pHRYlTQo=; __rmsid=-uP_84oNTQO0uWOrWj5frQ; _ym_isad=2; _ym_visorc=b; domain_sid=3T_uI-T9wPA6Z4GjBAqy3%3A1738593707545; tmr_detect=0%7C1738593718503',
        'Referer': 'https://www.rbc.ru/search/?query=%D1%81%D0%B1%D0%B5%D1%80%D0%B1%D0%B0%D0%BD%D0%BA&dateFrom=01.01.2025&dateTo=31.01.2025',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Chromium";v="130", "YaBrowser";v="24.12", "Not?A_Brand";v="99", "Yowser";v="2.5"',
        'sec-ch-ua-mobile': '?1',
        'sec-ch-ua-platform': '"Android"',
    }
    start = vrem1(start)
    finish = vrem1(finish)
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
        response = requests.get('https://www.rbc.ru/search/ajax/', params=params,headers=headers)
        vr = response.json()
        for i in range(len(vr['items'])):
            if (zapros in (vr['items'][i]['title'].split()) or (vr['items'][i]['body']!=None and zapros in (vr['items'][i]['body'].split()))) \
                    or (vr['items'][i]['project_nick'] != 'realty' and vr['items'][i]['category'] != 'Общество'
                        and vr['items'][i]['project_nick'] != 'rbcplus'):
                da['items'].append(vr['items'][i])
        if not vr['moreExists']:
            break
        if vr['moreExists']:
            nst += 1
    eli = pd.DataFrame(da['items'])[['title', 'publish_date_t']]
    eli['publish_date_t'] = eli['publish_date_t'].apply(lambda x: datetime.fromtimestamp(x))
    eli['time'] = eli['publish_date_t'].apply(lambda x: str(x)[11:])
    eli['publish_date_t'] = eli['publish_date_t'].apply(lambda x: str(x)[:10])
    eli.rename(columns={'publish_date_t': 'date'}, inplace=True)
    eli['source'] = 'RBK'
    return eli


def RiaNews(zapros, start, finish):
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
    ele['date'] = ele['date'].apply(lambda x: vrem2(x))
    ele['time'] = ele['time'].apply(lambda x: x + ':00')
    return ele


def repka(s):
    s = s[4:-5]
    s = s.replace('</span>', '')
    s = s.replace('<span class="allocation_found">', '')
    return s


def fim(s):
    s = s[s.find('>') + 1:-4]
    return s





zap = input()
na = input()
kon = input()
def AiFNews(zapros, start, finish):
    q = 1
    headers = {
        'x-requested-with': 'XMLHttpRequest',
    }
    da = {'title': [],
          'date': []
          # 'type': [],
          }
    while True:
        data = {
            'page': f'{q}',
        }
        response = requests.post(f'https://aif.ru/search/index/index/from/{na}/to/{kon}/text/{zap}',headers=headers,
                data=data)
        vr = response.json()
        sta = vr['isFinished']
        soup = BeautifulSoup(vr['data'], features="lxml")
        block = soup.findAll(class_='list_item')
        for item in block:
            da['title'].append(repka(str(item.find('h3'))))
            da['date'].append(str(item.find(class_='text_box__date'))[29:-7])
            # da['type'].append(fim(str(item.find(class_="rubric_link no_title_element_js"))))
        if not sta:
            q += 1
        if sta:
            break
        elo = pd.DataFrame(da)
        elo['time'] = elo['date'].apply(lambda x: x[11:] + ':00')
        elo['date'] = elo['date'].apply(lambda x: vrem2(x[:10]))
        elo['source'] = 'AiF'
        return elo


el = LentaNews(zap, na, kon)
eli = RbkNews(zap, na, kon)
elo = AiFNews(zap, na, kon)
btt = obed(el, eli, elo)
pd.set_option('display.max_rows', None)
print(btt)
# print(btt[['date','title','source']])
# print(eli[['date','time','source']])
# сбербанк
# 2025-01-01
# 2025-01-18
