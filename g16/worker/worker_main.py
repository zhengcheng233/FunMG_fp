import sys
import time
from pathlib import Path  # noqa: E402

module_g16_dir = Path(Path(__file__).parent.parent)
sys.path.append(module_g16_dir.as_posix())

from worker.typings import InputType  # noqa: E402
from job import run_fp,post_fp,Params  # noqa: E402
from loguru import logger  # noqa: E402
from worker.utils import acquire_task, update_task_state, TaskUpdateData, TaskOut  # noqa: E402
from datetime import datetime  # noqa: E402


data_prefix = "/vepfs/fs_users/chensq/project/funmg/runtime_data/tasks/dft"
log_path = "/vepfs/fs_users/chensq/project/funmg/runtime_data/tasks/dft/log/runtime.log"  # to be improve

logger.add("log.log", backtrace=True, diagnose=True)




def work_flow(task:TaskOut, path_prefix:Path):
    try:
        # 参数转换
        params = Params(**task.cfg.cfg)

        # 运行任务
        run_fp(params,path_prefix)

        update_task_state(
            task_id=task.id,
            data=TaskUpdateData(progress=95), ## 粗略设置快完成
        )

        # 后处理, 包括生成json文件或者光谱数据
        post_fp(params,path_prefix)
        return True
    except Exception as e:
        logger.error(f'dft calculation warning: {e}')
        return False


def worker():
    logger.info("worker init running...")
    count = 0
    current_task_id = ""
    while True:
        try:
            if count % 1000 == 1:
                logger.info(f"第{count}次主动获取任务")
            time.sleep(5)
            count = count + 1

            # 获取任务
            task = acquire_task(["QC_Calculator"])
            if not task:
                time.sleep(10)
            else:
                current_task_id = task.id
                # 更新任务启动时间和状态
                update_task_state(
                    task_id=task.id,
                    data=TaskUpdateData(
                        exec_time=datetime.now().timestamp(), state="running"
                    ),
                )
                
                
                # 主流程
                logger.info("on before run work flow")
                res = work_flow(task, Path(data_prefix,task.id))
                logger.info("on after run work flow")

                # 更新任务结束时间和状态
                update_task_state(
                    task_id=task.id,
                    data=TaskUpdateData(
                        end_time=datetime.now().timestamp(),
                        state="finished" if res else "failed",
                    ),
                )

        except Exception as e:
            # 更新任务结束时间和状态
            if current_task_id:
                update_task_state(
                    task_id=current_task_id,
                    data=TaskUpdateData(
                        end_time=datetime.now().timestamp(), state="failed"
                    ),
                )
            logger.error(f"error: task fail process {e};{current_task_id}")


if __name__ == "__main__":
    worker()
