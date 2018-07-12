#include <iostream>
#include <sys/stat.h>
#include <thread>
#include <deque>
#include <mutex>

#include <gtest/gtest.h>

static constexpr const char* CMAKE_PATH = "/usr/local/bin/cmake";
static constexpr const char* TMP_BUILD_DIR = "tmp_build";
static constexpr size_t NB_THREADS = 1;


class CompileFixture : public ::testing::Test {
public:
    CompileFixture() {
        std::cout << "" << std::endl;
        std::stringstream ss;
        ss << "pwd && mkdir " << TMP_BUILD_DIR;
        int status = run(ss);
        EXPECT_EQ(0, status);
    }

    void SetUp() {
        // code here will execute just before the test ensues
    }

    void TearDown() {
        // code here will be called just after the test completes
        // ok to throw exceptions from here if need be
    }

    ~CompileFixture() noexcept {
        std::stringstream ss;
        ss << "rm -rf " << TMP_BUILD_DIR;
        int status = run(ss);
        if(status != 0) {
            std::cout << "An error occurred while cleaning up build directory. Manual removal is needed." << std::endl;
        }
    }

    void runCmake(size_t i) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            std::cout << "thread id=" << i << ", formulas[i]=" << std::get<0>(formulas[i % formulas.size()])
                      << ", option=" << std::get<2>(formulas[i % formulas.size()]) << std::endl;
        }

        std::stringstream ss;
        ss << "cd " << TMP_BUILD_DIR << " && "
           << CMAKE_PATH << " -DCMAKE_BUILD_TYPE=Debug ../../keops -DFORMULA_OBJ=\"" << std::get<0>(formulas[i % formulas.size()])
           << "\" -DVAR_ALIASES=\"" << std::get<1>(formulas[i % formulas.size()]) << "\" "
           << std::get<2>(formulas[i % formulas.size()]) << "&& make -j1 keops";
        int status = run(ss);
        std::cout << "status=" << status << std::endl;
        ASSERT_EQ(0, status);

        std::lock_guard<std::mutex> lock(mutex);
        ++nbSuccessfulCompiles;
    }

    size_t nbSuccessfulCompiles = 0;

private:
    using myTuple = std::tuple<const char*, const char*, const char*>;
    const std::vector<myTuple> formulas = {
            myTuple("Exp(-p*SqNorm2(x-y))", "auto p=Pm(0,1); auto x=Vx(1,3); auto y=Vy(2,3);", "-D__TYPE__=float -Dshared_obj_name=keops0"),
            myTuple("Exp(-p*SqNorm2(x-y))", "auto p=Pm(0,1); auto x=Vx(1,3); auto y=Vy(2,3);", "-D__TYPE__=double -Dshared_obj_name=keops1"),
            myTuple("Square(p-a)*Exp(x+y)", "auto p=Pm(0,1); auto a=Vy(1,1); auto x=Vx(2,3); auto y=Vy(3,3);", "-D__TYPE__=float -Dshared_obj_name=keops2"),
            myTuple("Square(p-a)*Exp(x+y)", "auto p=Pm(0,1); auto a=Vy(1,1); auto x=Vx(2,3); auto y=Vy(3,3);", "-D__TYPE__=double -Dshared_obj_name=keops3"),
            myTuple("Exp(-G*SqDist(X,Y)) * P", "auto G=Pm(0,1); auto X=Vx(1,3); auto Y=Vy(2,3); auto P=Vy(3,3);", "-D__TYPE__=float -Dshared_obj_name=keops4"),
            myTuple("Exp(-G*SqDist(X,Y)) * P", "auto G=Pm(0,1); auto X=Vx(1,3); auto Y=Vy(2,3); auto P=Vy(3,3);", "-D__TYPE__=double -Dshared_obj_name=keops5")
    };
    std::mutex mutex;

    int run(const std::stringstream& ss) noexcept {
        return system(ss.str().c_str());
    }
};





TEST_F(CompileFixture, concurrentThreads) {

    std::deque<std::thread> threads;

    for(uint8_t i=0; i<NB_THREADS; ++i) {
        threads.emplace_front(std::thread(&CompileFixture::runCmake, this, i));
    }

    for(uint8_t i=0; i<NB_THREADS; ++i) {
        threads[i].join();
    }

    ASSERT_EQ(NB_THREADS, nbSuccessfulCompiles);

//    runCmake(1);

}



