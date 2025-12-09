-- Citibike Star Schema

-- NYC Citibike Analytics for TheCommons XR



CREATE SCHEMA IF NOT EXISTS citibike;



-- Dimension: Date

CREATE TABLE citibike.dim_date (

    date_id SERIAL PRIMARY KEY,

    full_date DATE NOT NULL,

    day_num INTEGER NOT NULL,           -- 1-7 (first week of month)

    day_name VARCHAR(20) NOT NULL,      -- Monday, Tuesday, etc.

    month_num INTEGER NOT NULL,

    month_name VARCHAR(20) NOT NULL,    -- July, November

    year INTEGER NOT NULL,

    quarter_num INTEGER NOT NULL,

    holiday_name VARCHAR(50)

);



-- Dimension: Bike Type

CREATE TABLE citibike.dim_bike_type (

    bike_type_id SERIAL PRIMARY KEY,

    bike_type_name VARCHAR(20) NOT NULL  -- classic_bike, electric_bike

);



-- Dimension: Time of Day

CREATE TABLE citibike.dim_time_of_day (

    time_of_day_id SERIAL PRIMARY KEY,

    time_of_day_name VARCHAR(20) NOT NULL,  -- Morning, Afternoon, Evening, Night

    start_hour INTEGER NOT NULL,

    end_hour INTEGER NOT NULL

);



-- Dimension: Member Type

CREATE TABLE citibike.dim_member_type (

    member_type_id SERIAL PRIMARY KEY,

    member_type_name VARCHAR(20) NOT NULL  -- member, casual

);



-- Fact: Rides

CREATE TABLE citibike.fact_rides (

    ride_id SERIAL PRIMARY KEY,

    start_station_name VARCHAR(255),

    end_station_name VARCHAR(255),

    date_id INTEGER REFERENCES citibike.dim_date(date_id),

    bike_type_id INTEGER REFERENCES citibike.dim_bike_type(bike_type_id),

    member_type_id INTEGER REFERENCES citibike.dim_member_type(member_type_id),

    time_of_day_id INTEGER REFERENCES citibike.dim_time_of_day(time_of_day_id),

    started_at TIMESTAMP,

    ended_at TIMESTAMP

);



-- Indexes for query performance

CREATE INDEX idx_fact_rides_date ON citibike.fact_rides(date_id);

CREATE INDEX idx_fact_rides_bike_type ON citibike.fact_rides(bike_type_id);

CREATE INDEX idx_fact_rides_time ON citibike.fact_rides(time_of_day_id);

CREATE INDEX idx_fact_rides_stations ON citibike.fact_rides(start_station_name, end_station_name);



-- Seed dimension data

INSERT INTO citibike.dim_bike_type (bike_type_name) VALUES ('classic_bike'), ('electric_bike');



INSERT INTO citibike.dim_time_of_day (time_of_day_name, start_hour, end_hour) VALUES 

    ('Morning', 6, 12),

    ('Afternoon', 12, 17),

    ('Evening', 17, 21),

    ('Night', 21, 6);



INSERT INTO citibike.dim_member_type (member_type_name) VALUES ('member'), ('casual');

