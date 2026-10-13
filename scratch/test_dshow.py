import sys
import ctypes
from ctypes import wintypes, POINTER, c_void_p, c_wchar_p, c_ulong, Structure, byref, HRESULT

def list_dshow_names():
    ole32 = ctypes.oledll.ole32
    oleaut32 = ctypes.oledll.oleaut32
    
    # Init COM
    hr = ole32.CoInitialize(None)
    
    class GUID(Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8)
        ]
        def __init__(self, d1, d2, d3, d4):
            super().__init__(d1, d2, d3, (wintypes.BYTE * 8)(*d4))

    CLSID_SystemDeviceEnum = GUID(0x62BE5E31, 0x60EB, 0x11d0, [0xBD, 0x3B, 0x00, 0xA0, 0xC9, 0x11, 0xCE, 0x86])
    CLSID_VideoInputDeviceCategory = GUID(0x860BB310, 0x5D01, 0x11d0, [0xBD, 0x3B, 0x00, 0xA0, 0xC9, 0x11, 0xCE, 0x86])
    IID_ICreateDevEnum = GUID(0x29840822, 0x5B84, 0x11D0, [0xBD, 0x3B, 0x00, 0xA0, 0xC9, 0x11, 0xCE, 0x86])
    IID_IPropertyBag = GUID(0x55272A00, 0x42CB, 0x11CE, [0x81, 0x35, 0x00, 0xAA, 0x00, 0x4B, 0xB8, 0x51])

    class VARIANT(Structure):
        _fields_ = [
            ("vt", wintypes.WORD),
            ("wReserved1", wintypes.WORD),
            ("wReserved2", wintypes.WORD),
            ("wReserved3", wintypes.WORD),
            ("bstrVal", c_void_p)
        ]

    # Let's see if we can do this simply
    print("Testing DShow COM enumeration...")

list_dshow_names()
