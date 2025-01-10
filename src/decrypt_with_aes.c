// Decrypt Data With AES

BOOL AESDecrypt(char *pPassword, char *pData, DWORD pDataLen, DWORD *pOutputDataLen)
{
  BOOL lRetVal = FALSE;
  HCRYPTPROV lCryptProvHandle = 0;
  HCRYPTKEY lKeyHandle = 0;
  HCRYPTHASH lHashHandle = 0;
 
  if (CryptAcquireContext(&lCryptProvHandle, NULL, NULL, PROV_RSA_AES, 
      CRYPT_VERIFYCONTEXT))
  {
    if (CryptCreateHash(lCryptProvHandle, CALG_SHA_256, 0, 0, &lHashHandle))
    {
      if (CryptHashData(lHashHandle, (PBYTE) pPassword,(DWORD) strlen(pPassword), 0))
      {
        if (CryptDeriveKey(lCryptProvHandle, CALG_AES_256, lHashHandle, 
            CRYPT_EXPORTABLE, &lKeyHandle))
        {	
          if (CryptDecrypt(lKeyHandle, 0, TRUE, 0, (BYTE *) pData, &pDataLen))
          {
            *pOutputDataLen = pDataLen;
            lRetVal = TRUE;
          }
          else
            *pOutputDataLen = 0;
 
          CryptDestroyKey(lKeyHandle);
        } // if (CryptDeriveKey(...
      } // if (CryptHashData(...
      CryptDestroyHash(lHashHandle);
    } // if (CryptCreateHash(...
    CryptReleaseContext(lCryptProvHandle, 0);
  } // if (CryptAcquireContext(...
 
  return(lRetVal);
}
