from datetime import datetime, timedelta
import holidays
import pandas as pd


class Especificacion_dias():

#Clase  constructor 

    def __init__(self):
        self.lista_fechas=[]
        self.lista_lunes = []
        self.lista_martes = []
        self.lista_miercoles = []
        self.lista_jueves = []
        self.lista_viernes = []
        self.lista_sabado = []
        self.lista_domingo = []
        self.lista_festivos = []

    def days_to_list(self, day, argument):

    # Entradas:
    # - Recibe days que es de tipo datatime 
    # - Recibe un argumento que es un numero relacionado a el numero del dia de la semana del datetime analizado.

    # Proceso:
    # - Prepara el metodo para que se pueda ejecutar segun el numero de dia.

    # Salida:
    # - Ejecuta el metodo necesario segun el numero del dia.

        method_name = 'day_' + str(argument)
        method = getattr(self, method_name)
        return method(day)
    
    def day_0(self, day):

    # Entradas:
    # - Recibe un dia tipo lunes tipo datetime.

    # Salida:
    # - Devuelve la lista de los lunes que hay hasta el momento.

        return self.lista_lunes.append(day.strftime('%Y-%m-%d'))
    
    def day_1(self, day):

    # Entradas:
    # - Recibe un dia tipo martes.

    # Salida:
    # - Devuelve la lista de los martes que hay hasta el momento.

        return self.lista_martes.append(day.strftime('%Y-%m-%d'))

    def day_2(self, day):

    # Entradas:
    # - Recibe un dia tipo miercoles.

    # Salida:
    # - Devuelve la lista de los miercoles que hay hasta el momento.

        return self.lista_miercoles.append(day.strftime('%Y-%m-%d'))

    def day_3(self, day):

    # Entradas:
    # - Recibe un dia tipo jueves.

    # Salida:
    # - Devuelve la lista de los jueves que hay hasta el momento.

        return self.lista_jueves.append(day.strftime('%Y-%m-%d'))

    def day_4(self, day):

    # Entradas:
    # - Recibe un dia tipo viernes.

    # Salida:
    # - Devuelve la lista de los viernes que hay hasta el momento.

        return self.lista_viernes.append(day.strftime('%Y-%m-%d'))

    def day_5(self, day):

    # Entradas:
    # - Recibe un dia tipo sabado.

    # Salida:
    # - Devuelve la lista de los sabados que hay hasta el momento.

        return self.lista_sabado.append(day.strftime('%Y-%m-%d'))

    def day_6(self, day):

        # Entradas:
        # - Recibe un dia tipo domingo.

        # Salida:
        # - Devuelve la lista de los domingos que hay hasta el momento.

        return self.lista_domingo.append(day.strftime('%Y-%m-%d'))

    def get_week(self, list_dates):

        # Entradas:
        # -  recibe una lista de dias tipo datetime.

        # Proceso:
        # - Recorre la lista de los detetime ingresados y obtiene cual es el numero del dia, guardandolo en el diccionario. 

        # Salida:
        # - Devuelve el diccionario con las lista correspondientes a las llaves tipo lunes, martes, miercoles, jueves, viernes y los clasifica como (01234)
        try:
            for day in list(list_dates):
                if self._get_festivos_col(day):
                    self.lista_festivos.append(day.strftime('%Y-%m-%d'))
                    for date, name in sorted(holidays.CO(years=2021).items(), reverse=True):
                        if day.date() > date:
                            if len(self.lista_festivos) < 6:
                                self.lista_festivos.append(date.strftime('%Y-%m-%d'))
                    list_dates.remove(day)
                else:
                    self.days_to_list(day, day.weekday())
            dic = dict(zip('01234567', [self.lista_lunes,
                                        self.lista_martes,
                                        self.lista_miercoles,
                                        self.lista_jueves,
                                        self.lista_viernes,
                                        self.lista_sabado,
                                        self.lista_domingo,
                                        self.lista_festivos]))
            return dic
        except:
            print('no se pudieron crear diccionario de semanas (dic_week)')

    def get_days(self, date, historic_days):

    # Entradas
    # - Este metodo recibe el date que es una fecha pero en tipo String.
    # -  Recibe un historic_days que es de tipo int
    # Proceso
    # - obtiene la fecha tipo date y la fecha final tipo date y ejecuta la _get_days y obtiene la lista de los dias teniendo encuenta las consideraciones. 
    # Salida
    # - Retorna la lista tipo date que se tiene que analizar 
        try:
            date = datetime.strptime(date, '%Y-%m-%d')
            past_date = date - timedelta(historic_days)
            lista_fechas = [past_date + timedelta(days=d) for d in range ((date - past_date).days + 1)]
            return lista_fechas
        except:
                print('no se pudieron crear la lista de dias  (lista)')
                print('fecha ingresada ',date)
                print('dias hacia atras ',historic_days)
    
    def _get_festivos_col(self, day):

    #  Entradas:
    # - Recibe un datetime. 

    # Proceso:
    # - Verifica si la fecha es un dia festivo.

    # Salida:
    # - devuelve un falso o verdadero si el datetime es festivo o no.

        col_holidays = holidays.CountryHoliday('CO')
        return day in col_holidays
    
    
    def get_intervalos(self, minutos):

        # Entradas:
        # - Recibe los minutos que debe tener cada intervalo.

        # Proceso:
        # - Obtiene todos los intervalos desde la fecha inicial hasta la fecha final 

        # Salida:       
        # - Restorna un dataframe procesado.

        try:

            inicio = datetime(2021, 12, 13, 00, 00, 00)
            actual = datetime(2021, 12, 13, 23, 59, 59)
            df = pd.DataFrame()
            df.append({'t_intervalo': inicio.strftime('%H:%M:%S')}, ignore_index=True)
            fg = True
            con = 1
            while(fg):
                intervalo = inicio + timedelta(minutes=minutos*con)
                df  = df.append({'t_intervalo': intervalo.strftime('%H:%M:%S')}, ignore_index=True)
                if intervalo > actual:
                    fg=False
                con +=1
            df['t_intervalo'] = df['t_intervalo'].astype(str)
            df.drop_duplicates().sort_values(by=['t_intervalo'], ascending= True)
            return df
        
        except:
            print('no se pudieron crear intervalos')
            if minutos > 0 :
                print('el valor del intervalo es minutos= ',minutos)