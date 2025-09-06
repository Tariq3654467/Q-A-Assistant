from rag import build_vectorstore

if __name__ == "__main__":
    print("Building vectorstore from ./data ...")
    build_vectorstore("../data")
    print("Done. Index saved to ./backend/storage")
