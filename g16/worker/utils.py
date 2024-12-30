import time
import base64
from loguru import logger
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from worker.typings import WorkerInfo,TaskOut,AppId,WorkerUpdateTaskData,TaskUpdate,TaskUpdateData,List
import httpx


logger.add("worker.log", format="{time} {level} {message}", level="DEBUG", backtrace=True, diagnose=True, rotation="500 MB", )


def encrypt_timestamp(password:str):
    # 生成密钥
    salt = b'salt_'  # 使用一个固定的盐，或者随机生成
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=10000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    # 创建Fernet实例
    f = Fernet(key)
    
    # 获取当前时间戳并加密
    current_timestamp = str(int(time.time())).encode()
    encrypted_timestamp = f.encrypt(current_timestamp)
    
    return encrypted_timestamp.decode()

def decrypt_and_validate(encrypted_string:str, password:str):
    # 生成密钥
    salt = b'salt_'  # 使用与加密时相同的盐
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=10000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    # 创建Fernet实例
    f = Fernet(key)
    
    try:
        # 解密时间戳
        decrypted_timestamp = f.decrypt(encrypted_string.encode())
        original_timestamp = int(decrypted_timestamp.decode())
        
        # 获取当前时间戳
        current_timestamp = int(time.time())
        
        # 检查时间间隔
        time_difference = current_timestamp - original_timestamp
        
        if time_difference < 10:
            return True
    except Exception as e:
        logger.error(f"task deliver error: unknown worker attempt to get task. {e}")
        return False
    

def acquire_task(app_ids:List[AppId], url:str='https://funmg.dp.tech/api/task/consume_task'):
        # url=f"http://{host}:{port}/task/consume_task",
        # app_ids=['OLED_spectrum']
        try:
            token=encrypt_timestamp('worker@hal9k')
            worker_info=WorkerInfo(
                worker_id='dev_funmg',
                app_ids=app_ids,
                token=token
            )
            res = httpx.post(
                url=url,
                json=worker_info.dict()
            )
            res.raise_for_status()
            res_data = res.json()
            if res_data['code'] != 0 :
                logger.error(f"获取任务出错: {res_data['msg']}, {worker_info.dict()}")
                return None
    
            if not res_data['data']:
                logger.info(f"task is {res_data['data']},msg:{res_data['msg']}")
                return None
            return TaskOut(**res_data['data'])
        except Exception as e:
            logger.error(f"获取任务出错: {e}")
            return None
        
def update_task_state(task_id:str,data:TaskUpdateData,url:str='https://funmg.dp.tech/api/task/update_by_worker'):
    ## url=f"http://{host}:{port}/task/update_by_worker",
    try:
        token=encrypt_timestamp('worker@hal9k')
        worker_update_data=WorkerUpdateTaskData(
            token=token,
            worker_id='dev_funmg',
            data=TaskUpdate(
                id = task_id,
                data=data
            )
        )
        res = httpx.post(
            url=url,
            json= worker_update_data.dict()
        )
        res.raise_for_status()
        res_data = res.json()
        if res_data['code'] != 0 :
            logger.error(f"更新任务状态出错: {res_data['msg']}, {worker_update_data.dict()}")
            return False

        logger.info(f"success update task {task_id} state: {data}")
        return True

    except Exception as e:
        logger.error(f"更新任务状态出错: {e}, {worker_update_data.dict()}")
        return False




if __name__ == '__main__':


    # 使用示例
    encrypted = encrypt_timestamp('worker')
    print("Encrypted timestamp:", encrypted)

    # # 等待一些时间（例如，5秒）
    # time.sleep(5)

    valid = decrypt_and_validate(encrypted,"worker")
    print("Validation result:", valid)