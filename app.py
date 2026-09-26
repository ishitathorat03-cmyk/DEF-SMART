import io
import re
import hashlib
from collections import defaultdict

import fitz
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Defence News Scanner OCR', page_icon='🛡️', layout='wide')
st.title('🛡️ Defence News Scanner OCR')
st.caption('Layout-aware OCR for extracting complete defence-related newspaper articles')

OCR_CONFIG = '--oem 3 --psm 4'
OCR_DPI = 220
MIN_ARTICLE_WORDS = 18
MIN_BODY_WORDS = 12

VERY_STRONG_PHRASES = [
    'indian army','indian navy','indian air force','armed forces','corps commander','corps commanders',
    'military operation','military operations','military exercise','military exercises','military deployment',
    'military deployments','military talks','air defence','air defense','missile strike','missile strikes',
    'missile attack','missile attacks','drone strike','drone strikes','drone attack','drone attacks',
    'naval exercise','naval exercises','defence ministry','defense ministry','ministry of defence',
    'ministry of defense','border security','border forces','troop deployment','special forces',
    'fighter aircraft','fighter jet','fighter jets','warship','warships','airbase','air base','submarine',
    'submarines','artillery unit','military unit','military commander','military commanders','military personnel',
    'security forces','line of actual control','line of control','loc','lac'
]
STRONG_TERMS = [
    'army','navy','military','missile','missiles','drone','drones','troop','troops','soldier','soldiers',
    'commander','commanders','brigade','battalion','regiment','artillery','warship','frigate','submarine',
    'aircraft','fighter','helicopter','airbase','defence','defense','border','combat','weapon','weapons',
    'ammunition','paramilitary','special forces','corps','irgc','air force','naval','operation','operations',
    'forces','deployment','deploy','war','wars','attack','attacks','strike','strikes'
]
NON_DEFENCE_TERMS = [
    'bharatanatyam','dance','dancer','singer','music','film','movie','actor','actress','cricket','football',
    'sports','education','student','students','college','school','jobs','employment','business','economy',
    'economic','hospital','hostel','rescue','rescued','volcano','earthquake','flood','weather','culture',
    'cultural','election','elections','political party','coalition government'
]

def normalize_text(text):
    text = (text or '').replace('\u00ad','').replace('ﬁ','fi').replace('ﬂ','fl')
    return re.sub(r'\s+',' ',text).strip()

def clean_text(text):
    text = normalize_text(text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([\(\[]) +', r'\1', text)
    text = re.sub(r'\s+([\)\]])', r'\1', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    return text.strip()

def words(text): return re.findall(r"\b[\w’'-]+\b", text.lower())
def word_count(text): return len(words(text))
def contains_phrase(text, phrase): return re.search(r'\b'+re.escape(phrase)+r'\b', text) is not None

def term_occurrences(text, terms):
    return sum(len(re.findall(r'\b'+re.escape(t)+r'\b', text)) for t in terms)

def make_hash(text):
    return hashlib.md5(re.sub(r'\W+','',text.lower())[:5000].encode('utf-8')).hexdigest()

def defence_analysis(text):
    text = clean_text(text).lower()
    if word_count(text) < MIN_ARTICLE_WORDS: return False, [], 0
    phrase_hits = [p for p in VERY_STRONG_PHRASES if contains_phrase(text,p)]
    term_hits = [t for t in STRONG_TERMS if contains_phrase(text,t)]
    non_hits = [t for t in NON_DEFENCE_TERMS if contains_phrase(text,t)]
    first_part = ' '.join(words(text)[:90])
    first_hits = sum(3 for p in VERY_STRONG_PHRASES if contains_phrase(first_part,p))
    first_hits += sum(1 for t in STRONG_TERMS if contains_phrase(first_part,t))
    defence_occurrences = term_occurrences(text, STRONG_TERMS)
    score = len(phrase_hits)*6 + len(term_hits) + first_hits - len(non_hits)*2
    if len(non_hits) >= 2 and len(phrase_hits) == 0: return False, phrase_hits+term_hits, score
    if len(phrase_hits) >= 1 and defence_occurrences >= 4 and score >= 8: return True, phrase_hits+term_hits, score
    if len(term_hits) >= 4 and defence_occurrences >= 6: return True, phrase_hits+term_hits, score
    if first_hits >= 4 and len(term_hits) >= 2 and defence_occurrences >= 4: return True, phrase_hits+term_hits, score
    return False, phrase_hits+term_hits, score

def extract_native_blocks(page):
    out=[]
    try: data=page.get_text('dict')
    except Exception: return out
    for block in data.get('blocks',[]):
        if block.get('type') != 0: continue
        x0,y0,x1,y1=block.get('bbox',[0,0,0,0]); lines=[]; sizes=[]
        for line in block.get('lines',[]):
            parts=[]
            for span in line.get('spans',[]):
                txt=normalize_text(span.get('text',''))
                if txt:
                    parts.append(txt)
                    try: sizes.append(float(span.get('size',0)))
                    except Exception: pass
            if parts: lines.append(' '.join(parts))
        text=clean_text(' '.join(lines))
        if word_count(text)>=2:
            out.append({'x0':float(x0),'y0':float(y0),'x1':float(x1),'y1':float(y1),'text':text,
                        'font_size':sum(sizes)/len(sizes) if sizes else 0.0})
    return out

def render_page(page,dpi=OCR_DPI):
    pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False)
    return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')

def ocr_page(image):
    data=pytesseract.image_to_data(image,config=OCR_CONFIG,output_type=pytesseract.Output.DICT)
    out=[]
    for i,txt0 in enumerate(data.get('text',[])):
        txt=normalize_text(txt0)
        if not txt: continue
        try: conf=float(data['conf'][i])
        except Exception: conf=-1
        if conf<20: continue
        x=int(data['left'][i]); y=int(data['top'][i]); w=int(data['width'][i]); h=int(data['height'][i])
        out.append({'x0':x,'y0':y,'x1':x+w,'y1':y+h,'text':txt,'font_size':0.0})
    return out

def blocks_to_lines(blocks):
    if not blocks:return []
    items=sorted(blocks,key=lambda b:(b['y0'],b['x0'])); lines=[]
    for item in items:
        cy=(item['y0']+item['y1'])/2; h=max(1,item['y1']-item['y0']); best=None
        for line in lines[-8:]:
            if abs(line['cy']-cy)<=max(5,h*0.65): best=line; break
        if best is None:
            lines.append({'x0':item['x0'],'y0':item['y0'],'x1':item['x1'],'y1':item['y1'],'cy':cy,'height':h,
                          'text_parts':[item['text']],'font_sizes':[item.get('font_size',0.0)]})
        else:
            best['x0']=min(best['x0'],item['x0']); best['y0']=min(best['y0'],item['y0']); best['x1']=max(best['x1'],item['x1']); best['y1']=max(best['y1'],item['y1'])
            best['height']=max(best['height'],h); best['cy']=(best['y0']+best['y1'])/2; best['text_parts'].append(item['text']); best['font_sizes'].append(item.get('font_size',0.0))
    final=[]
    for line in lines:
        text=clean_text(' '.join(line['text_parts'])); sizes=[s for s in line['font_sizes'] if s>0]
        if word_count(text)>=1:
            final.append({'x0':line['x0'],'y0':line['y0'],'x1':line['x1'],'y1':line['y1'],'height':line['height'],
                          'font_size':sum(sizes)/len(sizes) if sizes else 0,'text':text})
    return sorted(final,key=lambda x:(x['y0'],x['x0']))

def detect_columns(lines,page_width):
    if not lines:return []
    centers=sorted([((l['x0']+l['x1'])/2,l) for l in lines],key=lambda z:z[0]); tol=max(35,page_width*0.035); clusters=[]
    for center,line in centers:
        if not clusters: clusters.append([center,[line]]); continue
        if abs(center-clusters[-1][0])<=tol:
            clusters[-1][1].append(line); vals=[(x['x0']+x['x1'])/2 for x in clusters[-1][1]]; clusters[-1][0]=sum(vals)/len(vals)
        else: clusters.append([center,[line]])
    large=[c for c in clusters if len(c[1])>=3]
    if not large:return [sorted(lines,key=lambda x:(x['y0'],x['x0']))]
    columns=[sorted(c[1],key=lambda x:(x['y0'],x['x0'])) for c in sorted(large,key=lambda c:c[0])]
    if len(columns)>8:
        buckets=defaultdict(list); bw=page_width/6
        for line in lines:
            idx=max(0,min(5,int(((line['x0']+line['x1'])/2)/max(1,bw)))); buckets[idx].append(line)
        columns=[sorted(buckets[k],key=lambda x:(x['y0'],x['x0'])) for k in sorted(buckets)]
    return columns

def looks_like_headline(line,all_lines):
    text=line['text']; wc=word_count(text)
    if wc<3 or wc>35:return False
    alpha=re.sub(r'[^A-Za-z]','',text)
    if not alpha:return False
    upper=sum(c.isupper() for c in alpha)/max(1,len(alpha)); max_font=max([x.get('font_size',0) for x in all_lines]+[1])
    font_big=line.get('font_size',0)>=max_font*.88 and line.get('font_size',0)>=10
    return upper>=.72 or font_big or (wc<=16 and line['height']>14)

def split_column_into_articles(lines):
    if not lines:return []
    lines=sorted(lines,key=lambda x:x['y0']); gaps=[max(0,b['y0']-a['y1']) for a,b in zip(lines,lines[1:])]; normal=max(2,sum(sorted(gaps)[:max(1,len(gaps)//2)])/max(1,len(sorted(gaps)[:max(1,len(gaps)//2)]))) if gaps else 4
    candidates=[]; current=[]
    for line in lines:
        if not current: current=[line]; continue
        prev=current[-1]; gap=max(0,line['y0']-prev['y1']); current_words=word_count(' '.join(x['text'] for x in current))
        large_gap=gap>max(28,normal*4.5)
        new_headline=looks_like_headline(line,lines) and current_words>=MIN_ARTICLE_WORDS and line['y0']>prev['y0']
        if large_gap or new_headline:
            candidates.append(current); current=[line]
        else: current.append(line)
    if current:candidates.append(current)
    return [c for c in candidates if word_count(' '.join(x['text'] for x in c))>=MIN_ARTICLE_WORDS]

def extract_articles_from_blocks(blocks,page_number,source,page_width):
    lines=blocks_to_lines(blocks)
    if not lines:return []
    articles=[]
    for ci,col in enumerate(detect_columns(lines,page_width),1):
        for candidate in split_column_into_articles(col):
            text=clean_text(' '.join(x['text'] for x in candidate))
            ok,signals,score=defence_analysis(text)
            if ok and word_count(text)>=MIN_BODY_WORDS:
                articles.append({'page':page_number,'column':ci,'source':source,'text':text,'signals':signals,'score':score,'words':word_count(text)})
    return articles

def crop_columns_from_image(image):
    width,height=image.size; n=6; overlap=int(width*.025); sw=width/n; crops=[]
    for i in range(n):
        x0=int(max(0,i*sw-overlap)); x1=int(min(width,(i+1)*sw+overlap))
        if x1-x0>=100:crops.append((i+1,image.crop((x0,0,x1,height))))
    return crops

def extract_ocr_column_articles(image,page_number):
    out=[]
    for ci,crop in crop_columns_from_image(image):
        found=extract_articles_from_blocks(ocr_page(crop),page_number,'OCR',crop.size[0])
        for a in found:a['column']=ci
        out.extend(found)
    return out

def remove_duplicates(articles):
    if not articles:return []
    unique=[]; seen=set()
    for a in sorted(articles,key=lambda x:(x['page'],-x['words'],-x['score'])):
        h=make_hash(a['text'])
        if h in seen:continue
        prefix=set(' '.join(words(a['text'])[:80]).split()); duplicate=False
        for old in unique:
            if old['page']!=a['page']:continue
            oldset=set(' '.join(words(old['text'])[:80]).split())
            overlap=len(prefix&oldset)/max(1,len(prefix|oldset))
            if overlap>=.72:duplicate=True;break
        if not duplicate:seen.add(h);unique.append(a)
    return sorted(unique,key=lambda a:(a['page'],a['column']))

def process_page(page,page_number):
    native=extract_native_blocks(page); native_wc=word_count(' '.join(x['text'] for x in native)); native_articles=[]
    if native_wc>=30:native_articles=extract_articles_from_blocks(native,page_number,'Native PDF',float(page.rect.width))
    if native_articles:return native_articles,'Native PDF'
    image=render_page(page); ocr_articles=extract_ocr_column_articles(image,page_number)
    if ocr_articles:return ocr_articles,'OCR'
    full=extract_articles_from_blocks(ocr_page(image),page_number,'OCR',image.size[0])
    return full,('Native PDF' if native_wc>=30 else 'OCR')

def scan_pdf(pdf_bytes,selected_pages,update_progress=None):
    doc=fitz.open(stream=pdf_bytes,filetype='pdf'); all_articles=[]; native_count=0; ocr_count=0
    total=len(selected_pages)
    for pos,pn in enumerate(selected_pages,1):
        try:
            arts,src=process_page(doc.load_page(pn-1),pn); all_articles.extend(arts)
            if src=='Native PDF':native_count+=1
            else:ocr_count+=1
        except Exception as exc:
            st.warning(f'Page {pn} could not be processed: {exc}')
        if update_progress:update_progress(pos/max(1,total))
    doc.close(); return remove_duplicates(all_articles),native_count,ocr_count

uploaded_file=st.file_uploader('Upload Newspaper PDF',type=['pdf'])
if uploaded_file is not None:
    pdf_bytes=uploaded_file.getvalue()
    try:
        temp=fitz.open(stream=pdf_bytes,filetype='pdf'); total_pages=len(temp); temp.close()
    except Exception as exc:
        st.error(f'Could not open PDF: {exc}'); st.stop()
    st.success(f'📄 {uploaded_file.name} | {total_pages} pages')
    st.subheader('Select scan range')
    option=st.radio('Pages',['First 3 pages - TEST','All pages','Custom range'],horizontal=True)
    if option=='First 3 pages - TEST': selected_pages=list(range(1,min(3,total_pages)+1))
    elif option=='All pages': selected_pages=list(range(1,total_pages+1))
    else:
        start_page,end_page=st.slider('Select pages',1,total_pages,(1,min(3,total_pages))) if total_pages>1 else (1,1)
        selected_pages=list(range(start_page,end_page+1))
    st.info(f'Selected pages: {len(selected_pages)}')

    if st.button('🔍 Scan Newspaper',type='primary',use_container_width=True):
        if not selected_pages:
            st.warning('Select at least one page.'); st.stop()
        progress=st.progress(0); status=st.empty(); status.info('Reading newspaper layout...')
        def update_progress(value):
            value=min(max(value,0.0),1.0); progress.progress(value)
            status.info(f"Scanning newspaper: {int(value*len(selected_pages))}/{len(selected_pages)} pages...")
        try:
            articles,native_count,ocr_count=scan_pdf(pdf_bytes,selected_pages,update_progress)
        except Exception as exc:
            progress.empty(); status.empty(); st.error('The newspaper scan failed.'); st.exception(exc); st.stop()
        progress.progress(1.0); status.success('Newspaper scan completed!')
        st.session_state['articles']=articles
        st.session_state['scan_stats']=(len(selected_pages),native_count,ocr_count)

    articles=st.session_state.get('articles',[])
    stats=st.session_state.get('scan_stats')
    if stats:
        st.divider(); st.subheader('📊 Scan Summary'); m1,m2,m3,m4=st.columns(4)
        with m1:st.metric('Pages scanned',stats[0])
        with m2:st.metric('Native PDF',stats[1])
        with m3:st.metric('OCR',stats[2])
        with m4:st.metric('Defence articles',len(articles))
    if articles:
        st.divider(); st.subheader('📰 Defence News Found'); st.write(f'{len(articles)} defence-related article(s) detected.')
        rows=[]
        for i,a in enumerate(articles,1):
            st.markdown(f'### News {i}')
            st.caption(f"📄 Page {a['page']} • Column {a['column']} • {a['source']} • {a['words']} words")
            if a['signals']:st.caption('🔎 Defence signals: '+', '.join(a['signals'][:12]))
            st.write(a['text']); st.divider()
            rows.append({'News':i,'Page':a['page'],'Column':a['column'],'Source':a['source'],'Defence signals':', '.join(a['signals']),'Article':a['text']})
        csv=pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig')
        st.download_button('⬇️ Download Defence News CSV',csv,'defence_news_scanned.csv','text/csv',use_container_width=True)
    elif stats is not None:
        st.divider(); st.info('No genuine defence-related articles were detected in the selected pages.')
else:
    st.info('Upload an English newspaper PDF to begin scanning.')
