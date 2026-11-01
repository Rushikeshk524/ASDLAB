#Aim: Create a DAG named simple_dag with two Python tasks, task_1 and task_2, where task_1 executes before task_2. Schedule the DAG to run daily, execute it using the Airflow web interface, and monitor the task status and logs.

# Theory: Apache Airflow is a workflow management tool used to schedule and monitor tasks. A DAG (Directed Acyclic Graph) named simple_dag is created with two Python tasks named task_1 and task_2. PythonOperator is used to define the tasks. The dependency task_1 >> task_2 ensures that task 1 executes before task 2. The DAG is scheduled to run daily and its execution is monitored using the Airflow web interface.

# Result(Optional): The simple_dag was successfully created with two Python tasks. Both tasks were executed in sequence, and their execution status and logs were monitored through the Airflow web interface.  

# conclusion(Optional): Thus, a simple workflow was successfully created and executed using Apache Airflow, demonstrating task scheduling, execution and monitoring.

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def task1():
    print("Task 1 executed")

def task2():
    print("Task 2 executed")

with DAG(
    dag_id='simple_dag',
    start_date=datetime(2026, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:

    t1 = PythonOperator(
        task_id='task_1',
        python_callable=task1
    )

    t2 = PythonOperator(
        task_id='task_2',
        python_callable=task2
    )

    t1 >> t2