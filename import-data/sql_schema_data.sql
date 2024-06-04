create schema machine_learning;
create table sampla_data_1(
	number1 int,
    number2 int
);

insert into sampla_data_1(number1, number2) values
(1,10),
(2,20),
(3,30),
(4,40),
(5,50),
(6,60),
(7,70),
(8,80),
(9,90),
(10,100);

select * from sampla_data_1;