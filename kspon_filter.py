import pandas as pd

def get_duration(file):
    #a,sr = librosa.load(file)
    #return len(a)
    return librosa.get_duration(filename=file)

if __name__ == '__main__':
    kspon_data = pd.read_csv('kspon_data.tsv',sep='\t')
    kspon_data['length'] = kspon_data['file'].apply(lambda x: get_duration(x))
    
    duration = sorted(list(kspon_data['length'].values))

    print('start measuring duration')
    print('0%:', str(duration[:10]))
    print('50%: ',str(duration[int(len(duration) * 0.5)]))
    print('80%: ',str(duration[int(len(duration) * 0.8)]))
    print('90%: ',str(duration[int(len(duration) * 0.9)]))
    print('95%: ',str(duration[int(len(duration) * 0.95)]))
    print('98%: ',str(duration[int(len(duration) * 0.98) -1]))
    print('99%: ',str(duration[int(len(duration) * 0.99) -1]))
    print('99.5%: ',str(duration[int(len(duration) * 0.995) -1]))
    print('100%: ',str(duration[int(len(duration)) -1]))

    print(len(duration) - int(len(duration) * 0.995))
    print(len(duration))

    stt2_kspon = [length for length in duration if length < 7.62]
    stt1_kspon = [length for length in duration if length * 16000 < 213347]

    print('stt1','*'*40 )
    print('start measuring duration')
    print('0%:', str(stt1_kspon[:10]))
    print('50%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.5)]))
    print('80%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.8)]))
    print('90%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.9)]))
    print('95%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.95)]))
    print('98%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.98) -1]))
    print('99%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.99) -1]))
    print('99.5%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.995) -1]))
    print('100%: ',str(stt1_kspon[int(len(stt1_kspon)) -1]))

    print(len(stt1_kspon) - int(len(stt1_kspon) * 0.995))
    print(len(stt1_kspon))

    print('stt2','*'*40 )
    print('start measuring duration')
    print('0%:', str(stt2_kspon[:10]))
    print('50%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.5)]))
    print('80%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.8)]))
    print('90%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.9)]))
    print('95%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.95)]))
    print('98%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.98) -1]))
    print('99%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.99) -1]))
    print('99.5%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.995) -1]))
    print('100%: ',str(stt2_kspon[int(len(stt2_kspon)) -1]))

    print(len(stt2_kspon) - int(len(stt2_kspon) * 0.995))
    print(len(stt2_kspon))
    
    kspon_data[kspon_data['length'] * 16000 < 213347].to_csv('stt1_kspon.tsv',sep='\t',index=False)
    kspon_data[kspon_data['length'] < 7.62].to_csv('stt2_kspon.tsv',sep='\t',index=False)
    