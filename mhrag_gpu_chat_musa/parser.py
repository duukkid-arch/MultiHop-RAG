"""PDF (PyMuPDF / pypdf / pdftotext), DOCX and plain text parsing."""
from __future__ import annotations
import io,os,subprocess,tempfile,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from .store import chunk

MAX_FILE=20*1024*1024
MAX_PAGES=150
MAX_CHUNKS=1800

def pdf_pages(content):
    errors=[]
    try:
        import fitz
        doc=fitz.open(stream=content,filetype='pdf')
        if len(doc)>MAX_PAGES:raise ValueError(f'PDF 超过 {MAX_PAGES} 页上限')
        try:return [(i+1,doc[i].get_text('text')) for i in range(len(doc))]
        finally:doc.close()
    except ImportError:errors.append('PyMuPDF 不可用')
    except ValueError:raise
    except Exception as exc:errors.append('PyMuPDF: '+str(exc)[:150])
    try:
        from pypdf import PdfReader
        doc=PdfReader(io.BytesIO(content))
        if len(doc.pages)>MAX_PAGES:raise ValueError(f'PDF 超过 {MAX_PAGES} 页上限')
        return [(i+1,p.extract_text() or '') for i,p in enumerate(doc.pages)]
    except ImportError: errors.append('pypdf 不可用')
    except ValueError:raise
    except Exception as exc:errors.append('pypdf: '+str(exc)[:150])
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf',delete=False) as temp:
            temp.write(content);path=temp.name
        try:
            p=subprocess.run(['pdftotext','-layout',path,'-'],capture_output=True,timeout=60,check=True)
            pages=p.stdout.decode('utf-8',errors='replace').split('\f')
            pages=[x for x in pages if x.strip()]
            if len(pages)>MAX_PAGES:raise ValueError(f'PDF 超过 {MAX_PAGES} 页上限')
            return list(enumerate(pages,1))
        finally:os.unlink(path)
    except FileNotFoundError:errors.append('pdftotext 不可用')
    except ValueError:raise
    except Exception as exc:errors.append('pdftotext: '+str(exc)[:150])
    raise ValueError('PDF 无法解析；可安装 PyMuPDF 或检查文件。'+ '; '.join(errors))

def docx_text(content):
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            raw=z.read('word/document.xml')
        root=ET.fromstring(raw)
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        lines=[]
        for p in root.findall('.//w:p',ns):
            txt=''.join(e.text or '' for e in p.findall('.//w:t',ns))
            if txt.strip():lines.append(txt)
        return '\n'.join(lines)
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError('DOCX 文件不是有效的 Word 文档') from exc

def extract(filename,content):
    suffix=Path(filename).suffix.lower()
    if not content:raise ValueError('文件为空')
    if len(content)>MAX_FILE:raise ValueError('单文件上限 20 MB')
    if suffix=='.pdf': pages=pdf_pages(content)
    elif suffix=='.docx':pages=[(None,docx_text(content))]
    elif suffix in ('.txt','.md','.markdown'):pages=[(None,content.decode('utf-8-sig',errors='replace'))]
    else:raise ValueError('仅支持 PDF / DOCX / TXT / MD')
    result=[]
    for page,text in pages:
        for piece in chunk(text):result.append((page,piece))
        if len(result)>MAX_CHUNKS:raise ValueError(f'文件解析后超过 {MAX_CHUNKS} 个段落')
    if not result or sum(len(s) for _,s in result)<40:
        raise ValueError('未提取到足够正文：可能是扫描 PDF，需要 OCR；请检查 PDF 是否可复制文字。')
    return result
