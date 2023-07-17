import numpy as np
import os
import sys
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
DIR = os.getcwd()
sys.path.insert(0, DIR)
#from streamlit_style import apply_style

# App title and header
st.set_page_config(page_title="Programs. ", page_icon=None, layout="wide", initial_sidebar_state="auto")
#apply_style()

######################################## tabs ########################################
listTabs =["🧑‍🏭Junior AI", "🧑‍🎓College Ready", "📈 Math In Programming", "🔢 On-demand", "        "]

whitespace = 9
st.markdown("#### 💡 Checkout our exciting programs!")
## Fills and centers each tab label with em-spaces
tabs = st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

with tabs[0]:
    st.markdown("##### Junior AI Program")
    style = """
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            padding-bottom: 0.5em;
        }
        .subtitle {
            font-size: 24px;
            font-weight: bold;
            padding-bottom: 0.5em;
        }
        ul {
            margin-top: 0.5em;
            margin-bottom: 0.5em;
        }
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)

    st.markdown("""
        <div class="title">AI Education Project Programs for Students</div>
        <div class="subtitle">Project-Driven Learning for Real-World Applications</div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <div class="subtitle">Goal:</div>
            <ul>
                <li>Introduce middle/high school students to the exciting world of artificial intelligence and machine learning.</li>
                <li>Inspire the next generation of AI innovators.</li>
                <li>Develop critical thinking, problem-solving, and teamwork skills.</li>
                <li>Prepare students for future challenges in the rapidly evolving tech industry.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <div class="subtitle">Who:</div>
            <ul>
                <li>Middle school students who are interested in technology and want to learn about the applications of AI.</li>
                <li>Open to all students, regardless of their previous experience with programming or AI.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <div class="subtitle">When We Teach:</div>
            <ul>
                <li>12-week program offered throughout the year.</li>
                <li>Flexible program that allows students to learn at their own pace and according to their own schedule.</li>
                <li>Students can choose to take the program during the school year or over the summer.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <div class="subtitle">How We Teach:</div>
            <ul>
                <li>Project-driven learning, where students learn by working on real-world projects.</li>
                <li>Experienced instructors guide students through the process of building AI models using popular tools such as Python, TensorFlow, and Keras.</li>
                <li>Incorporates interactive quizzes, games, and other activities to help reinforce concepts.</li>
                <li>Emphasizes safety and ethics when working with AI, covering topics such as data privacy, bias, and the ethics of AI.</li>
                <li>Teaches students how to safely use generative AI and chatbots, such as ChatGPT, while ensuring they understand the risks involved.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <div class="subtitle">What Will Students Learn:</div>
            <ul>
                <li>Fundamentals of AI, including supervised and unsupervised learning, deep learning, computer vision, and natural language processing.</li>
                <li>Valuable skills in problem-solving, critical thinking, and teamwork. In addition, they will have learned how to use popular AI tools, including TensorFlow and Keras, and how to safely use generative AI and chatbots.
            </ul>
        </div>
    """,  unsafe_allow_html=True)
    st.markdown("""
        <div class="section">
            <div class="subtitle">How We Teach:</div>
            <ul>
                <li>Our program is project-driven, which means that students learn by working on real-world projects. Our experienced instructors guide students through the process of building AI models using popular tools such as Python, TensorFlow, and Keras. We also incorporate interactive quizzes, games, and other activities to help reinforce the concepts that students learn.
                </li><li>We understand that safety and ethics are important considerations when working with AI. That's why we place a strong emphasis on teaching students how to use AI safely and responsibly. We cover topics such as data privacy, bias, and the ethics of AI. We also teach students how to safely use generative AI and chatbots, such as ChatGPT, while ensuring they understand the risks involved.</li>
            </ul>
        </div>
    """,  unsafe_allow_html=True)

    st.markdown("""
        <div class="section">
            <div class="subtitle">What Will Students Learn:</div>
            <ul>
                <li>By the end of the program, students will have gained a solid understanding of the fundamentals of AI, including supervised and unsupervised learning, deep learning, computer vision, and natural language processing. They will also have gained valuable skills in problem-solving, critical thinking, and teamwork. In addition, they will have learned how to use popular AI tools, including TensorFlow and Keras, and how to safely use generative AI and chatbots.
                </li><li>Our AI education program for middle school students is designed to give students the knowledge and skills they need to succeed in the tech industry of tomorrow. We believe that by inspiring the next generation of AI innovators, we can help shape a better future for everyone.</li>
            </ul>
        </div>
    """,  unsafe_allow_html=True)


#################################### college ready ####################################
with tabs[1]:
        st.markdown("##### College Ready")
        # Program Overview
        st.markdown("<h2 style='font-size: 36px; margin-bottom: 0;'>AI Education Program for High School Students</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 24px; margin-top: 0;'>Program Overview</h3>", unsafe_allow_html=True)

        st.markdown("""
        Our AI Education Program for High School Students is designed to provide an engaging and challenging curriculum that will introduce students to the exciting world of artificial intelligence. Through a series of hands-on projects and activities, students will learn the fundamentals of AI and machine learning and gain practical experience using cutting-edge technologies.

        Our program is taught by experienced AI professionals who are passionate about sharing their knowledge and expertise with the next generation of innovators. With a focus on real-world applications and problem-solving, students will develop critical thinking skills and gain a competitive edge in college and beyond.

        Here are some key highlights of our program:
        """)

        # Program Details
        st.markdown("<h3 style='font-size: 24px;'>Program Details</h3>", unsafe_allow_html=True)

        # Goal
        st.markdown("<h4 style='font-size: 18px;'>Goal</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Introduce students to the fundamentals of AI and machine learning
        - Develop critical thinking skills and problem-solving abilities
        - Provide practical experience using cutting-edge AI technologies
        """)

        # Who
        st.markdown("<h4 style='font-size: 18px;'>Who</h4>", unsafe_allow_html=True)
        st.markdown("""
        - High school students who are interested in computer science and technology
        - Students who want to gain a competitive edge in college and beyond
        - Students who are passionate about innovation and problem-solving
        """)

        # When we teach
        st.markdown("<h4 style='font-size: 18px;'>When We Teach</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our program is offered on a rolling basis throughout the year
        - Students can enroll at any time and complete the program at their own pace
        - All materials and resources are available online, 24/7
        """)

        # How we teach
        st.markdown("<h4 style='font-size: 18px;'>How We Teach (Project-Driven)</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our program is project-driven, with a focus on real-world applications and problem-solving
        - Students will work on a series of hands-on projects and activities that will help them develop practical skills and gain experience with cutting-edge AI technologies
        - Projects are designed to be challenging and engaging, encouraging students to think creatively and work collaboratively
        """)

        # What will student learn
        st.markdown("<h4 style='font-size: 18px;'>What Will Students Learn</h4>", unsafe_allow_html=True)
        st.markdown("""
        - The fundamentals of AI and machine learning
        - Practical experience using cutting-edge AI technologies
        - Critical thinking and problem-solving skills
        - Collaboration and teamwork
        - Effective communication and presentation skills
        - A competitive edge in college and beyond
        """)

        # College Application
        st.markdown("<h4 style='font-size: 24px;'>College Application</h4>", unsafe_allow_html=True)
        st.markdown("""Our AI education program for high school students provides an excellent opportunity for students to explore AI technologies and their real-world applications.
        By participating in our program, students will develop hands-on experience in developing AI solutions, learn how to think critically about AI's impact on society,
        and prepare themselves for college and future careers in AI and related fields. We invite all high school students who are interested in AI to join our program
        and take the first step towards becoming AI experts.""")

#################################### on-demand ####################################
with tabs[3]:
        st.markdown("##### Re-charge! On-deman")
        # Program Overview
        st.markdown("<h2 style='font-size: 36px; margin-bottom: 0;'>On-demand, Per-Request Training and Consulting Program</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 24px; margin-top: 0;'>Program Overview</h3>", unsafe_allow_html=True)

        st.markdown("""
        At our on-demand training program, we understand that everyone's learning needs are unique. That's why we offer a customizable and flexible approach to learning. With our program, you can choose the technology you want to learn and our experts will teach you based on your specific needs.
        """)

        # Program Details
        st.markdown("<h3 style='font-size: 24px;'>How it works:</h3>", unsafe_allow_html=True)

        # Goal
        st.markdown("<h4 style='font-size: 18px;'>Goal</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our on-demand training program is designed to help you stay ahead of the curve in the rapidly-evolving field of technology, and to give you the skills and knowledge you need to succeed in your career.
        - Whether you are looking to advance in your current job or explore new career opportunities, our program can help you achieve your goals.
        - By participating in our program, you will join a community of like-minded learners who are passionate about technology and committed to lifelong learning.
        """)

        # Who
        st.markdown("<h4 style='font-size: 18px;'>Who</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our on-demand training program is designed for individuals who are interested in expanding their knowledge and skills in the field of technology, including AI and other state-of-the-art technologies.
        - Whether you are a student, a working professional, or simply someone who wants to stay ahead of the curve, our program is perfect for anyone who wants to learn and grow in their career.
        - Our program is open to anyone, regardless of their prior experience or education level.
        """)

        # When we teach
        st.markdown("<h4 style='font-size: 18px;'>When We Teach</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our program is offered on a rolling basis throughout the year
        - Students can enroll at any time and complete the program at their own pace
        - All materials and resources are available online, 24/7
        """)

        # what
        st.markdown("<h4 style='font-size: 18px;'>What</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our program offers a wide range of topics to choose from, including AI, automation, main stream technologies, and expert help and consulting.
        - With our program, you can choose which topic you want to learn and bid on the price you are willing to pay. Our experts will then work with you to design a personalized training program that meets your specific needs and goals.
        - You will have access to high-quality learning materials, including videos, tutorials, and hands-on projects, all designed to help you master the topic of your choice.
        """)

        # how
        st.markdown("<h4 style='font-size: 18px;'>How</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our program is entirely on-demand, which means you can learn at your own pace and on your own schedule.
        - You will have access to our team of expert instructors, who will provide personalized guidance and support throughout your learning journey.
        - Our program is designed to be interactive and engaging, with plenty of opportunities for hands-on learning and collaboration with other learners.
        """)

        # on-demand
        st.markdown("<h4 style='font-size: 24px;'>On-demand, per-request or consulting</h4>", unsafe_allow_html=True)
        st.markdown("""At our on-demand training program, we are committed to helping individuals and businesses succeed in the rapidly changing world of technology.
        Whether it's AI, automation, web development, mobile app development, or expert help and consulting, we have the expertise and resources to help our customers achieve their learning goals.""")



#################################### math and programming ####################################
with tabs[2]:
        st.markdown("##### Math in Programming")
        # Program Overview
        st.markdown("<h2 style='font-size: 36px; margin-bottom: 0;'>Programming and Math Together for K-12 Students</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 24px; margin-top: 0;'>Program Overview</h3>", unsafe_allow_html=True)

        st.markdown("""
        Our program combines the power of technology, programming, and math to create a unique learning experience for K-12 students. Small class sizes and flexible scheduling ensure that every student gets the attention and support they need to succeed.
        """)

        # Program Details
        st.markdown("<h3 style='font-size: 24px;'>How it works:</h3>", unsafe_allow_html=True)


        # Who
        st.markdown("<h4 style='font-size: 18px;'>Who</h4>", unsafe_allow_html=True)
        st.markdown("""
        - The program is designed for all K-12 students who are interested in developing their programming and math skills.
        - Students with no prior experience in programming or math are welcome to join the program.
        - Our instructors are experienced in teaching students of all ages and skill levels.
        """)

        # What
        st.markdown("<h4 style='font-size: 18px;'>What</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our program combines programming and math to create a unique learning experience that will help students develop problem-solving, critical thinking, and analytical skills.
        - We cover topics such as coding fundamentals, algorithms, data structures, and mathematical concepts such as algebra, geometry, and calculus.
        - Our program uses project-based learning to engage students and help them apply their skills to real-world problems.
        """)

        # How
        st.markdown("<h4 style='font-size: 18px;'>How</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Our program is designed to be flexible and adaptable to each student's needs and interests.
        - We offer small class sizes, which allow our instructors to provide personalized attention and support to each student.
        - Our program is offered both online and in-person, with a range of scheduling options to fit any student's schedule.
        """)

        # how
        st.markdown("<h4 style='font-size: 18px;'>Why</h4>", unsafe_allow_html=True)
        st.markdown("""
        - We believe that programming and math are essential skills for the future and can provide students with many opportunities for personal and professional growth.
        - By combining programming and math, we help students develop a unique skill set that will set them apart in a competitive job market.
        - Our program provides a supportive and encouraging learning environment where students can explore their interests and reach their full potential.
        """)

        # on-demand
        st.markdown("<h4 style='font-size: 24px;'>Math Is Programming? or Programming Is Math?</h4>", unsafe_allow_html=True)
        st.markdown("""Our Tech Programming and Math Together program is designed to provide K-12 students with a unique and engaging learning experience. By combining programming and math, we aim to foster critical thinking, problem-solving, and creativity skills in our students. Our small class sizes and flexible schedule allow students to receive personalized attention and learn at their own pace. Our experienced instructors guide students through hands-on projects and real-world applications, giving them the skills and knowledge they need to succeed in a rapidly evolving tech industry.

Whether your child is just starting out in their programming and math journey or looking to take their skills to the next level, our program has something to offer. We believe that by inspiring the next generation of tech innovators, we can help shape a better future for everyone.""")
