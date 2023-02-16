
import os

def getLinkFromCollectionAndToken(collectionAddress, tokenId):
    return f"https://opensea.io/assets/ethereum/{collectionAddress}/{tokenId}"


def makeFolderIfNotExists(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

